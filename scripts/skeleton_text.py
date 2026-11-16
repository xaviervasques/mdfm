"""The structure and labels of the skeleton catalogue page (scripts/render_skeleton_catalogue.py), in English.

Owner decision of 2026-11-03: the project is published, so everything the tree produces for a reader is in English.
This module replaces the French tables of tasks SKELV to SKELV4b (scripts/skeleton_fr.py, removed) and holds only what
the clip specifications do not carry: the groups of the page's list in clinical order (GROUP_ORDER, GROUP_NAMES) and
their headings (SUBGROUPS), the series a clip belongs to (series_key for the exam set, cat_series_key for the symptom
catalogue) and their labels (SERIES_NAMES, CAT_SERIES_NAMES), the names of the exam's scale items (ITEM_NAMES, the
MDS-UPDRS part III titles), the grade chips and list lines of a series (series_chips, series_summary), the order of
text grades (TEXT_GRADE_RANK) and the per-time texts of a clip (overlays). The operator, the grade text and the
parameter text of a clip are the specification's own (exam.ClipSpec operator, grade_text, params_text or the text of
exam.describe), and the per-time texts are the truth's own words; nothing here rewrites them. Numbers keep their
decimal point. The words come from the catalogue's own vocabulary (docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md: its
families and rows) and the scale names as the specifications write them.

page_texts(meta) never raises: a catalogue series without a label takes its clip's operator and a clip without a
heading goes under 'Other'; both are returned as notes, which the page script prints and writes to its manifest
('unlabelled'), never a refusal.

Wave SH1d (the clinical review of the English page): the catalogue page reads each text through clinical_texts, which
takes the marks of provenance out of it (split_provenance: '[module]', '(module level)', '(D4)', 'declared map',
'estimator floor', '(cited normal)' ...) and lists them on a provenance line, each with the phrase it qualifies and
what it means (PROVENANCE_KINDS); the specifications' texts themselves are unchanged. The simple kinetic tremor
during pronation-supination has its TETRAS heading; STANDALONE_LEVEL0 names the level 0 clips kept on the page
without a symptom clip.
"""
import re

CATALOGUE_SINCE = "catalogue_v1"


def is_catalogue(meta):
    return meta.get("since") == CATALOGUE_SINCE


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


# ================================================================ the exam set (--set exam, skeleton_v1)
FAMILY_NAMES = {"tremor": "Tremor", "bradykinesia": "Bradykinesia", "freezing": "Freezing of gait",
                "dystonia": "Dystonia (postures)", "reference": "Healthy references"}
FAMILY_ORDER = ("tremor", "bradykinesia", "freezing", "dystonia", "reference")
MDS_UPDRS_ITEMS = {   # the titles of the MDS-UPDRS part III items the clips read
    "3.4": "Finger tapping", "3.5": "Hand movements", "3.6": "Pronation-supination movements of hands",
    "3.7": "Toe tapping", "3.8": "Leg agility", "3.9": "Arising from chair", "3.10": "Gait",
    "3.11": "Freezing of gait", "3.12": "Postural stability", "3.13": "Posture",
    "3.14": "Global spontaneity of movement (body bradykinesia)", "3.15": "Postural tremor of the hands",
    "3.16": "Kinetic tremor of the hands", "3.17": "Rest tremor amplitude", "3.18": "Constancy of rest tremor"}
ITEM_NAMES = {   # the headings of the exam clips (their scale item); an item not listed is its own heading
    **{k: f"{k} {v}" for k, v in MDS_UPDRS_ITEMS.items()},
    "3.10 reference": "3.10 Gait", "reference": "Standing", "level 0": "Level 0 clips (neutral body)",
    "level": "Starting postures (degrees)"}
EXAM_VIEWS = ("front", "back", "left", "right", "top", "hands", "hands_top", "foot_left", "head", "track_left",
              "examiner", "hand_left", "hand_left_medial")
#   the views of the exam page (task SKELV): viewer.VIEWS grows, the exam page keeps these and the regions they frame


def series_key(clip_id):
    """The symptom an exam clip belongs to: its id without the grade token _g<grade>."""
    return re.sub(r"_g\d+[ab]?(?=_|$)", "", clip_id)


SERIES_NAMES = [  # (regex on an exam series key, label), first match
    (r"^tremor_E1_rest$", "rest, left forearm"), (r"^tremor_E1_constancy$", "constancy (at 3.17 grade 2)"),
    (r"^tremor_E2_postural$", "postural, left forearm"), (r"^tremor_E2_reemergent$", "re-emergent"),
    (r"^tremor_E2_raise$", "arms raised to the posture, level 0"),
    (r"^tremor_E2_raise_long_rest$", "arms raised after a longer rest, level 0"),
    (r"^tremor_E2_raise_postural$", "arms raised, postural"),
    (r"^tremor_E2_raise_reemergent$", "arms raised, re-emergent"),
    (r"_level0$", "level 0"), (r"_decrement$", "amplitude decrement"), (r"_interruptions$", "interruptions"),
    (r"_arrest$", "arrest"), (r"_slowing$", "slowing (proposed ladder)"),
    (r"^freezing_E8_trembling$", "trembling in place"), (r"^freezing_E8_akinesia$", "akinesia"),
    (r"^ref_walk_E8$", "healthy walk (gait generator)"), (r"^ref_stand_E7$", "healthy stance (procedural)"),
    (r"^ref_arise_E7$", "healthy arising from a chair (procedural)"),
    (r"^dystonia_E7_level0$", "standing, level 0"), (r"^dystonia_E1_level0$", "seated, level 0"),
    (r"neck_rotation_left_duration3$", "rotation to the left, intermittent (duration score 3)"),
    (r"neck_rotation_left$", "rotation to the left"), (r"neck_tilt_left$", "tilt to the left"),
    (r"neck_anterocollis$", "anterocollis"), (r"neck_retrocollis$", "retrocollis"),
    (r"shoulder_elevation_elevation$", "left shoulder elevation"),
    (r"trunk_lateral_left_duration(\d)$", r"lateral flexion to the left, grade 3, DIS duration \1"),
    (r"start_neck_tilt_left_20deg$", "neck, tilt to the left 20 degrees"),
    (r"start_trunk_lateral_left_15deg$", "trunk, lateral flexion to the left 15 degrees"),
    (r"start_distal_leg_knee_flexion_30deg$", "left knee, flexion 30 degrees"),
    (r"start_arm_abduction30_elbow100$", "left arm, abduction 30 and elbow 100 degrees"),
    (r"trunk_lateral_left$", "lateral flexion to the left"), (r"proximal_arm_abduction$", "left arm abduction"),
    (r"distal_arm_elbow_flexion$", "left elbow flexion"), (r"proximal_leg_abduction$", "left hip abduction"),
    (r"distal_leg_knee_flexion$", "left knee flexion"),
    (r"distal_leg_plantarflexion_ankle$", "left ankle plantarflexion (seated)"),
]
EXAM_V5_SERIES = {"tremor_E2_postural": "postural, left hand"}
#   the exam_v5 postural tremor reads the hand (its operator says 'left hand'); the skeleton_v1 data keep the forearm


def series_name(key, operator=""):
    """The label of an exam series (SERIES_NAMES; EXAM_V5_SERIES when the operator names the hand)."""
    if key in EXAM_V5_SERIES and "left hand" in (operator or ""):
        return EXAM_V5_SERIES[key]
    for pat, lab in SERIES_NAMES:
        m = re.search(pat, key)
        if m:
            return m.expand(lab)
    raise KeyError(f"no label for the series {key}")


FLOOR_MARKS = {"a": " (above the estimator floor)", "b": " (below the estimator floor)"}
#   the two levels of a grade 1 (ids _g1a, _g1b), in the specifications' words (exam.py, tremors.py)


def grade_chip(meta):
    """The short label of an exam clip's grade for the grade selector."""
    m = re.search(r"_g(\d+)([ab]?)(?=_|$)", meta["clip_id"])
    if meta["grade"] == "level":
        deg = re.findall(r"\d+(?:\.\d+)?", meta["grade_text"])
        return " and ".join(deg) + " degrees"
    if m:
        sub = FLOOR_MARKS.get(m.group(2), "")
        return f"{m.group(1)}{sub}"
    return str(meta["grade"])


def segment_name(meta):
    """The title of a clip's exam segment: the clip's own (exam.segment_title, written into its data), else its
    segment code."""
    return meta.get("segment_title") or meta["segment"]


def page_texts(meta):
    """({operator, grade, params, family, item, segment, series, series_key, chip} of a clip, notes). The operator,
    grade and parameter texts are the clip's own. A catalogue clip goes to catalogue_texts (its chip set per series
    by series_chips); never raises."""
    if is_catalogue(meta):
        return catalogue_texts(meta)
    notes = []
    key = series_key(meta["clip_id"])
    try:
        series = series_name(key, meta["operator"])
    except KeyError:
        series = meta["operator"]
        notes.append(f"series: {key}")
    return {"operator": meta["operator"], "grade": meta["grade_text"], "params": meta["params_text"] or "",
            "family": FAMILY_NAMES.get(meta["family"], _cap(meta["family"])),
            "item": ITEM_NAMES.get(meta["item"], meta["item"]), "segment": segment_name(meta), "series": series,
            "series_key": key, "chip": grade_chip(meta)}, notes


# ================================================================ the symptom catalogue (--set catalogue, skeleton_v2)
# ---------------------------------------------------------------- groups of the page's list, in clinical order
GROUP_ORDER = ("tremor", "reach", "bradykinesia", "parkinsonism", "atypical_parkinsonism", "parkinsonian_posture",
               "chorea", "dyskinesia", "stereotypy", "mirror", "myoclonus", "negative_myoclonus", "tics", "startle",
               "clonus", "dystonia", "paroxysmal", "ataxia", "freezing", "gait", "reference", "other")
GROUP_NAMES = {"tremor": "Tremors", "reach": "Finger-to-nose test (dysmetria)", "bradykinesia": "Bradykinesia",
               "parkinsonism": "Parkinsonism, other signs", "atypical_parkinsonism": "Atypical parkinsonism (PSP)",
               "parkinsonian_posture": "Parkinsonian postures (camptocormia, Pisa syndrome, antecollis)",
               "chorea": "Chorea, ballism, athetosis", "dyskinesia": "Dyskinesias (levodopa-induced, tardive)",
               "stereotypy": "Stereotypies, akathisia, restless legs", "mirror": "Mirror movements",
               "myoclonus": "Myoclonus", "negative_myoclonus": "Negative myoclonus (asterixis)", "tics": "Tics",
               "startle": "Startle", "clonus": "Clonus", "dystonia": "Dystonia",
               "paroxysmal": "Paroxysmal dyskinesias", "ataxia": "Ataxia", "freezing": "Freezing of gait",
               "gait": "Gait", "reference": "Healthy references", "other": "Other"}
FAMILY_GROUP = {"tremor": "tremor", "reach": "reach", "bradykinesia": "bradykinesia", "myoclonus": "myoclonus",
                "negative_myoclonus": "negative_myoclonus", "tics": "tics", "startle": "startle", "clonus": "clonus",
                "dystonia": "dystonia", "freezing": "freezing", "reference": "reference", "gait": "gait",
                "parkinsonism": "parkinsonism", "parkinsonian_posture": "parkinsonian_posture", "chorea": "chorea",
                "ballism": "chorea", "athetosis": "chorea", "dyskinesia": "dyskinesia", "stereotypy": "stereotypy",
                "akathisia": "stereotypy", "restless_legs": "stereotypy", "moving_toes": "stereotypy",
                "mirror": "mirror", "cervical_dystonia": "dystonia", "limb_dystonia": "dystonia",
                "dystonia_distribution": "dystonia", "paroxysmal": "paroxysmal", "ataxia": "ataxia",
                "atypical parkinsonism": "atypical_parkinsonism", "functional": "myoclonus"}
SERIES_GROUP = {"tremor_E3_reach_left": "reach", "tremor_E3_reach_right": "reach",   # the reach's level 0s
                "jerk_clonus_J2": "clonus", "jerk_asterixis_E2": "negative_myoclonus",   # a level 0 beside its grades
                "invol_stereotypy_flapping": "stereotypy", "invol_mirror_E4a": "mirror",
                "orthostatic_myoclonus_E7": "myoclonus", "gait_E8_fog_shuffling": "freezing"}


def group_of(meta):
    """The group of a clip in the page's list: its series' group (SERIES_GROUP), else its family's (FAMILY_GROUP,
    every gait_* family to 'gait'), else 'other'."""
    key = cat_series_key(meta["clip_id"]) if is_catalogue(meta) else series_key(meta["clip_id"])
    if key in SERIES_GROUP:
        return SERIES_GROUP[key]
    fam = meta["family"]
    if fam.startswith("gait"):
        return "gait"
    return FAMILY_GROUP.get(fam, "other")


# ---------------------------------------------------------------- series of the catalogue clips
CAT_SERIES_RULES = [   # (regex, replacement) applied in order on a catalogue clip id: its id without the grade
    (r"_g\d+[ab]?(?=_|$)", ""),
    (r"_(?:tetras|sara|icars|scats)\d+(?:p5)?$", ""),
    (r"_f\d_[ad]\d$", ""), (r"_a\d$", ""), (r"_conn\d_sev\d$", ""),
    (r"_(?:posture_only|posture)$", ""), (r"_level0$", ""), (r"_present$", ""),
    (r"_(number|frequency|intensity|complexity)_[nfic]\d$", r"_\1"),
    (r"^(jerk_startle_E\d)_(?:normal|hyperekplexia)$", r"\1"),
    (r"^(gait_E8_steppage)_(?:left|bilateral)$", r"\1"),
    (r"^gait_E8_(?:trendelenburg|duchenne)_right$", "gait_E8_trendelenburg_right"),
    (r"^(gait_E8_dystonic_left)_l\d$", r"\1"),
    (r"^(gait_E8_functional)_.+$", r"\1"),
    (r"_a\d_d\d$", ""), (r"^(invol_(?:lid|td)_.+)_s\d$", r"\1"),
    (r"^invol_E7_flap$", "invol_stereotypy_flapping"), (r"^invol_E4a_mirror$", "invol_mirror_E4a"),
    (r"^(post_E7_(?:lumbar|thoracic|lateral|neck))_(?:mild|open)$", r"\1"),
    (r"^(post_E7_cd_rotation)_duration\d$", r"\1_duration"), (r"^(post_E7_cd_shoulder)_D\d_.+$", r"\1"),
    (r"^(post_E7_cd_geste)_C\d$", r"\1"), (r"_sev\d$", ""), (r"^(post_E7_pkd)_(?:dystonic|choreic)$", r"\1"),
    (r"^(park_E7_arise)_(?:slow|attempts)$", r"\1"), (r"^(park_E4a?L_sequence_(?:ft|ps))_level\d$", r"\1"),
    (r"^(park_E4aL_ulfreeze)_long$", r"\1"),
    (r"^(jerk_tics_E\d_touch)_c\d$", r"\1"), (r"^(gait_E8_spastic_diplegic)_mild$", r"\1"),
    (r"^(park_E22_sitdown)_enbloc$", r"\1"), (r"^(park_E1_motorstate)_(?:offon|onoff)$", r"\1"),
    (r"^(jerk_tonic_tics_E1)_f\d_i\d(?:_mixed)?$", r"\1"), (r"^(jerk_functional_E\d_axial)_sev\d_dur\d$", r"\1"),
    (r"^(tremor_E8_walk)_rest$", r"\1"),
    (r"^gait_E(?:8|19)_vestibular_left_eyes_(?:open|closed)$", "gait_E8_vestibular_left"),
    (r"^gait_E(?:8|19)_sensory_ataxic_eyes_(?:open|closed)$", "gait_E8_sensory_ataxic"),
]


def cat_series_key(clip_id):
    """The symptom a catalogue clip belongs to: its id without its grade tokens (CAT_SERIES_RULES)."""
    k = clip_id
    for pat, rep in CAT_SERIES_RULES:
        k = re.sub(pat, rep, k)
    return k


CAT_SERIES_NAMES = {   # the label of a catalogue series under its heading (the page title is 'heading: label')
    # tremors
    "tremor_E1_pillrolling": "left fingers, at rest",
    "tremor_E1_pillrolling_supination": "with pronation-supination, left hand",
    "tremor_E2_essential": "postural, both arms (TETRAS 4a)",
    "tremor_E3_essential": "kinetic, left arm, finger to nose (TETRAS 4c)",
    "tremor_E3_essential_terminal": "kinetic with a terminal crescendo (intention component), left arm, finger to "
                                    "nose (TETRAS 4c)",
    "tremor_E1_essential_rest": "at rest (essential tremor plus), both hands",
    "tremor_E1_head_pitch": "yes-yes, seated", "tremor_E1_head_yaw": "no-no, seated",
    "tremor_E7_head_pitch": "yes-yes, standing", "tremor_E7_head_yaw": "no-no, standing",
    "tremor_E1_head_mixed": "mixed (no-no and yes-yes), seated",
    "tremor_E3_kinetic": "left arm, finger to nose",
    "tremor_E4_kinetic_ps": "left arm, during pronation-supination",
    "tremor_E3_intention": "left arm, rising near the target (SARA item 6)",
    "tremor_E3_cerebellar_combined": "intention tremor and dysmetria, left arm (SARA items 5 and 6)",
    "tremor_E1_holmes": "at rest, left arm", "tremor_E2_holmes": "in posture, left arm",
    "tremor_E3_holmes": "at rest, in posture and in action, left arm",
    "tremor_E2_dystonic": "left arm, jerky and irregular",
    "tremor_E1_dystonic_head": "head, no-no, on a dystonic neck rotation to the left",
    "tremor_E2w_wing": "both arms, proximal (TETRAS 4b)",
    "tremor_E7_orthostatic": "legs, standing",
    "standing_tremor_E10": "TETRAS standing posture, knees bent",
    "isometric_tremor_E4b_fist": "left fist clenched, forearm supported",
    "tremor_E1_leg_rest": "left leg and foot, seated",
    "tremor_E8_walk": "left forearm, during a parkinsonian walk",
    "tremor_E2_physiological": "both arms (TETRAS 4a)",
    "tremor_E2f_functional": "left arm, entrained by the tapping of the right hand",
    # finger to nose
    "tremor_E3_reach_left": "left arm", "tremor_E3_reach_right": "right arm",
    "reach_E3_dysmetria_past": "past the target (hypermetria), left arm",
    "reach_E3_dysmetria_short": "short of the target (hypometria), left arm",
    "reach_E3_decomposition": "left arm",
    # myoclonus
    "jerk_myoclonus_E2_cortical_focal": "focal, left arm, in posture",
    "jerk_myoclonus_E1_multifocal": "multifocal (cortical)",
    "jerk_myoclonus_E1_generalized": "generalized (reticular)",
    "jerk_myoclonus_E1_segmental_right_arm": "segmental, rhythmic, right arm",
    "jerk_myoclonus_E7_propriospinal": "propriospinal, axial flexion, standing",
    "jerk_myoclonus_E1_propriospinal": "propriospinal, axial flexion, seated",
    "jerk_myoclonus_E2_stimulus": "arms outstretched, one jerk after each stimulus",
    "jerk_mdyst_E7_neck": "jerks over a dystonic neck rotation, standing",
    "jerk_mdyst_E1_right_arm": "jerks over a dystonic flexion of the right elbow, seated",
    "jerk_functional_E1_axial": "axial flexion, seated", "jerk_functional_E7_axial": "axial flexion, standing",
    "orthostatic_myoclonus_E7": "legs, standing",
    # negative myoclonus
    "jerk_asterixis_E2": "both hands, arms outstretched, wrists extended",
    "jerk_negative_E7_leg_lapses": "brief lapses of postural tone in the legs, standing",
    # tics
    "jerk_tics_E1_number": "seated, number of tics (YGTSS)",
    "jerk_tics_E1_frequency": "seated, frequency (MRVS)",
    "jerk_tics_E1_intensity": "seated, intensity (YGTSS)",
    "jerk_tics_E1_complexity": "seated, with orchestrated sequences",
    "jerk_tics_E7_complexity": "standing, orchestrated sequences and squat dips",
    "jerk_tics_E1_bouts_60s": "seated, bouts over one minute",
    "jerk_tics_E1_touch": "seated, touching the face (the right index, from the chair arm)",
    "jerk_tics_E7_touch": "standing, touching the face (the right index, from the hanging arm)",
    "jerk_tonic_tics_E1": "seated, held postures of the trunk and shoulder",
    # startle and clonus
    "jerk_startle_E7": "standing", "jerk_startle_E1": "seated",
    "jerk_clonus_J2": "left ankle, seated, the forefoot loaded",
    # neutral level 0s
    "jerk_E1": "seated", "jerk_E7": "standing",
    "jerk_E1_level0_60s": "seated, 60 s sequence (level 0 of the tics in bouts)",
    "invol_E1": "seated, hands on the thighs (level 0 of the involuntary movements)",
    "invol_E7": "standing (level 0 of the involuntary movements)",
    "invol_E2": "seated, arms outstretched (level 0 of the involuntary movements)",
    "invol_E1_level0_30s": "seated, hands on the thighs, longer sequence (level 0 of akathisia)",
    "invol_E7_level0_30s": "standing, longer sequence (level 0 of akathisia)",
    "invol_E1_level0_60s": "seated, hands on the thighs, long sequence (level 0 of the restless legs)",
    # gait
    "gait_E8": "parametric gait generator",
    "gait_E8_parkinsonian": "left side more affected (MDS-UPDRS 3.10)",
    "gait_E8_parkinsonian_festination": "festination",
    "gait_E8_parkinsonian_start_hesitation": "start hesitation (MDS-UPDRS 3.11)",
    "gait_E7p_retropulsion": "pull test, number of backward steps",
    "gait_E8_ataxic": "cerebellar (SARA item 1)",
    "gait_E8t_tandem": "tandem, heel to toe (SARA item 1)",
    "gait_E8_hemiparetic_left": "circumduction, left hemiparesis (Wernicke-Mann)",
    "gait_E8_spastic_diplegic": "spastic diplegic gait with scissoring",
    "gait_E8_steppage": "foot drop, left or bilateral",
    "gait_E8_trendelenburg_right": "right hip (Trendelenburg, Duchenne)",
    "gait_E8_waddling": "waddling gait (hip girdle weakness)",
    "gait_E8_antalgic_left": "left leg",
    "gait_E8_nph_magnetic": "normal-pressure hydrocephalus",
    "gait_E8_cautious": "cautious gait",
    "gait_E8_dystonic_left": "left foot and trunk",
    "gait_E8_choreic": "dancing gait",
    "gait_E8_functional": "clinical presentations",
    "gait_E8_psp": "wide base, stiff knees, neck extended",
    "gait_E8_dromedary": "dromedary gait (DYT1-like trunk dystonia)",
    "gait_E8_vestibular_left": "left vestibular lesion, eyes open then closed",
    "gait_E8_sensory_ataxic": "eyes open then closed",
    "gait_E8_fog_shuffling": "at the start, at the turn and in straight walking",
    # chorea, ballism, athetosis
    "invol_chorea_E1": "seated", "invol_chorea_E7": "standing",
    "invol_chorea_E1_trunk": "trunk, seated", "invol_chorea_E1_arms": "arms, seated",
    "invol_chorea_E1_legs": "legs, seated", "invol_chorea_E2_fingers": "both hands, arms outstretched",
    "invol_impersistence_E2": "arms outstretched", "invol_hemiballism_E1_left": "left arm and leg, seated",
    "invol_athetosis_E1": "hands, forearms, toes and neck, seated", "invol_choreoathetosis_E1": "seated",
    "invol_pseudoathetosis_E2_left": "left hand, arms outstretched, eyes closed",
    "invol_hd_dystonia_E7": "trunk and arms, on chorea, standing",
    # dyskinesias
    "invol_lid_peak_E1": "choreic, seated", "invol_lid_peak_E7": "choreic, standing",
    "invol_lid_dystonic_E1": "with choreic movements, seated",
    "invol_lid_diphasic_E1_kicking": "alternating kicking of the legs, seated",
    "invol_lid_diphasic_E7_stepping": "alternating stepping in place, standing",
    "invol_td_E1_rocking": "truncal rocking, seated", "invol_td_E1_limbs": "limbs, seated",
    "invol_td_E7_thrusting": "pelvic thrusting, standing",
    # stereotypies, akathisia, restless legs, moving toes, mirror movements
    "invol_stereotypy_rocking": "body rocking, seated", "invol_stereotypy_nodding": "head nodding, seated",
    "invol_stereotypy_flapping": "hand flapping, standing, hands before the chest",
    "invol_td_stereotypy_E7_marching": "marching in place, standing",
    "invol_td_stereotypy_E1_rubbing": "rubbing the thighs, seated",
    "invol_akathisia_E1": "seated", "invol_akathisia_E7": "standing", "invol_rls_E1": "seated",
    "invol_moving_toes_E1_left": "moving toes, left foot, seated",
    "invol_mirror_E4a": "left hand, during finger tapping of the right hand",
    # parkinsonian postures
    "post_E7_pd": "stooped posture, standing",
    "post_E7_lumbar": "lumbar fulcrum, standing", "post_E7_thoracic": "thoracic fulcrum, standing",
    "post_E7_lateral": "to the left, standing", "post_E7_neck": "standing",
    # dystonia
    "post_E7_cd_rotation_duration": "mobile rotation to the left, by duration score",
    "post_E7_cd_rotation_left_jerky": "rotation to the left with dystonic jerks",
    "post_E7_cd_tilt_left_jerky": "laterocollis to the left with dystonic jerks",
    "post_E7_cd_retrocollis_mobile": "mobile retrocollis",
    "post_E7_cd_shift_lateral": "lateral shift to the left",
    "post_E7_cd_shift_anterior": "anterior sagittal shift",
    "post_E7_cd_shift_posterior": "posterior sagittal shift",
    "post_E7_cd_shoulder": "left shoulder elevation",
    "post_E7_cd_torticollis_complex": "complex torticollis (rotation and tilt to the left, left shoulder "
                                      "elevation)",
    "post_E7_cd_geste": "geste antagoniste (right fingertips on the right cheek)",
    "post_W_writing": "writing, right hand, no dystonia (level 0)", "post_W_writers_cramp": "right hand",
    "post_E1_foot_dystonia": "left foot in equinovarus, seated",
    "post_E7_off_period_foot": "off-period dystonia of the left foot, standing",
    "post_E2_task_specific_arm": "left arm, arms outstretched",
    "post_E7_hemidystonia": "left hemidystonia (arm and leg)",
    "post_E7_segmental": "segmental dystonia (neck, left shoulder and arm)",
    "post_E7_multifocal": "multifocal dystonia (neck and left foot)",
    "post_E7_torsion_dystonia": "DYT1-like, at rest, standing",
    "post_W_overflow": "writer's cramp with overflow to the shoulder and neck",
    "post_W_writing_both": "writing with the left hand, the right resting (level 0)",
    "post_W_mirror": "right hand, while the left hand writes",
    "post_E7_opisthotonus": "sustained axial extension, standing",
    "post_E1_fixed_fist": "clenched left fist, seated", "post_E1_fixed_foot": "inverted left foot, seated",
    # paroxysmal dyskinesias
    "post_E7_pkd": "triggered by arising from a chair", "post_E1_pnkd_excerpt": "excerpt of an attack, seated",
    "post_E7_ped_excerpt": "excerpt of an attack, left foot, standing",
    # parkinsonism, other signs; atypical parkinsonism; Gowers' manoeuvre
    "park_E7_arise": "parkinsonian, arms folded",
    "park_E1_spontaneity": "seated", "park_E7_spontaneity": "standing",
    "park_E4aL_sequence_ft": "finger tapping, left hand", "park_E4L_sequence_ps": "pronation-supination, left hand",
    "park_E4a_ulfreeze": "finger tapping, left hand (MDS-UPDRS 3.4)",
    "park_E4aL_ulfreeze": "prolonged finger tapping, left hand",
    "park_E1_motorstate": "left hand, seated",
    "park_E7_arise_abrupt": "from a chair, arms folded", "park_E22_sitdown": "onto a chair, arms folded",
    "park_E21_applause": "seated, after the examiner's demonstration", "park_E7_gowers": "from a chair",
    # ataxia
    "atax_SARA2_natural": "natural stance", "atax_SARA2_together": "feet together",
    "atax_SARA2_tandem": "tandem stance (heel to toe)",
    "atax_ROMB": "healthy (level 0)", "atax_ROMB_cerebellar": "cerebellar ataxia",
    "atax_ROMB_sensory": "sensory ataxia", "atax_ROMB_sensory_bedside": "sensory ataxia, Romberg sign at the bedside",
    "atax_SARA3_sitting": "arms outstretched, back free", "atax_TITUB": "trunk and head, seated",
    "atax_SARA7_ddk": "left hand, on the chair arm",
    "atax_SARA8_heelshin": "left heel on the right shin, supine",
    "atax_SARA8_heelshin_right": "right heel on the left shin, supine (level 0 of the right side)",
}

# ---------------------------------------------------------------- headings inside a group
SUBGROUPS = [  # (key, heading, regex on the catalogue series key), in the order of the list
    ("pillrolling", "Pill-rolling rest tremor", r"^tremor_E1_pillrolling"),
    ("essential", "Essential tremor", r"^tremor_E\d_essential"),
    ("head", "Head tremor (TETRAS item 1)", r"^tremor_E\d_head_"),
    ("kinetic", "Kinetic tremor (MDS-UPDRS 3.16)", r"^tremor_E3_kinetic"),
    # wave SH1d (clinical review of the English page): the simple kinetic tremor during pronation-supination is
    # graded on the TETRAS upper-limb bands (its specification: no item of its own), not on 3.16 (finger to nose)
    ("kinetic_tetras", "Simple kinetic tremor (graded on the TETRAS upper-limb bands)", r"^tremor_E4_kinetic_ps"),
    ("intention", "Intention tremor (cerebellar)", r"^tremor_E3_(?:intention|cerebellar)"),
    ("holmes", "Holmes tremor", r"^tremor_E\d_holmes"),
    ("dystonic_tremor", "Dystonic tremor", r"^tremor_E\d_dystonic"),
    ("wing", "Wing-beating tremor", r"^tremor_E2w_wing"),
    ("orthostatic", "Orthostatic tremor", r"^tremor_E7_orthostatic"),
    ("standing_tremor", "Standing tremor of the legs and trunk (TETRAS item 9)", r"^standing_tremor_E10"),
    ("isometric", "Isometric tremor", r"^isometric_tremor_"),
    ("leg_rest", "Rest tremor of the leg", r"^tremor_E1_leg_rest"),
    ("walk_rest", "Rest tremor while walking", r"^tremor_E8_walk"),
    ("physiological", "Enhanced physiological tremor", r"^tremor_E2_physiological"),
    ("functional_tremor", "Functional tremor", r"^tremor_E2f_functional"),
    ("reach_level0", "Finger-to-nose test, level 0", r"^tremor_E3_reach_"),
    ("dysmetria", "Dysmetria (SARA item 5)", r"^reach_E3_dysmetria"),
    ("decomposition", "Decomposition of movement (ICARS)", r"^reach_E3_decomposition"),
    ("cortical", "Cortical action myoclonus (UMRS)", r"^jerk_myoclonus_E2_cortical"),
    ("rest_myoclonus", "Multifocal and generalized rest myoclonus (UMRS)",
     r"^jerk_myoclonus_E1_(?:multifocal|generalized)"),
    ("spinal", "Spinal myoclonus (segmental, propriospinal)", r"^jerk_myoclonus_E\d_(?:segmental|propriospinal)"),
    ("functional_jerks", "Functional axial jerks (with a dual task)", r"^jerk_functional_"),
    ("orthostatic_myoclonus", "Orthostatic myoclonus", r"^orthostatic_myoclonus_"),
    ("reflex", "Reflex myoclonus (stimulus-sensitive)", r"^jerk_myoclonus_E2_stimulus"),
    ("mdyst", "Myoclonus-dystonia", r"^jerk_mdyst_"),
    ("asterixis", "Asterixis (Conn; UMRS)", r"^jerk_asterixis_"),
    ("leg_lapses", "Negative myoclonus of the legs (UMRS)", r"^jerk_negative_"),
    ("simple_tics", "Simple motor tics (YGTSS, MRVS)", r"^jerk_tics_E1_(?:number|frequency|intensity)"),
    ("tonic_tics", "Tonic tics (dystonic)", r"^jerk_tonic_tics_"),
    ("complex_tics", "Complex motor tics", r"^jerk_tics_E\d_(?:complexity|touch)"),
    ("tic_bouts", "Tics in bouts", r"^jerk_tics_E1_bouts"),
    ("startle", "Startle reflex and hyperekplexia", r"^jerk_startle_"),
    ("clonus", "Ankle clonus (SCATS)", r"^jerk_clonus_"),
    ("gait_level0", "Healthy gait (level 0)", r"^gait_E8$"),
    ("parkinsonian_gait", "Parkinsonian gait", r"^gait_E8_parkinsonian"),
    ("psp_gait", "Gait of progressive supranuclear palsy (PSP)", r"^gait_E8_psp$"),
    ("postural_instability", "Postural instability (MDS-UPDRS 3.12)", r"^gait_E7p_retropulsion"),
    ("ataxic_gait", "Ataxic gait", r"^gait_E8t?_(?:ataxic|tandem)"),
    ("sensory_ataxic_gait", "Sensory ataxic gait (stamping)", r"^gait_E8_sensory_ataxic$"),
    ("vestibular_gait", "Vestibular gait with veering", r"^gait_E8_vestibular"),
    ("spastic_gait", "Spastic gaits (hemiparesis, diplegia)", r"^gait_E8_(?:hemiparetic|spastic)"),
    ("steppage", "Steppage gait", r"^gait_E8_steppage"),
    ("waddling", "Trendelenburg gait and waddling gait", r"^gait_E8_(?:trendelenburg|waddling)"),
    ("gowers", "Arising with Gowers' manoeuvre (hip girdle weakness)", r"^park_E7_gowers$"),
    ("antalgic", "Antalgic gait", r"^gait_E8_antalgic"),
    ("magnetic", "Magnetic gait", r"^gait_E8_nph"),
    ("cautious", "Cautious gait", r"^gait_E8_cautious"),
    ("dystonic_gait", "Dystonic gait (dromedary gait included)", r"^gait_E8_(?:dystonic|dromedary)"),
    ("choreic_gait", "Choreic gait", r"^gait_E8_choreic"),
    ("functional_gait", "Functional gait disorders", r"^gait_E8_functional"),
    ("arise", "Arising from chair (MDS-UPDRS 3.9)", r"^park_E7_arise$"),
    ("spontaneity", "Global spontaneity of movement (MDS-UPDRS 3.14)", r"^park_E\d_spontaneity$"),
    ("sequence", "Sequence effect (decrement)", r"^park_E4a?L_sequence_"),
    ("ul_freezing", "Upper-limb freezing", r"^park_E4aL?_ulfreeze$"),
    ("motor_fluctuations", "Motor fluctuations, ON and OFF (rest tremor)", r"^park_E1_motorstate$"),
    ("applause", "Applause sign", r"^park_E21_applause$"),
    ("arise_abrupt", "Abrupt arising", r"^park_E7_arise_abrupt$"),
    ("sitdown_enbloc", "Sitting down en bloc", r"^park_E22_sitdown$"),
    ("fog_shuffling", "Freezing with shuffling forward (small steps)", r"^gait_E8_fog_shuffling$"),
    ("pd_posture", "Posture (MDS-UPDRS 3.13)", r"^post_E7_pd$"),
    ("camptocormia", "Anterior trunk flexion and camptocormia (MDS consensus)", r"^post_E7_(?:lumbar|thoracic)$"),
    ("pisa", "Lateral trunk flexion and Pisa syndrome (MDS consensus)", r"^post_E7_lateral$"),
    ("antecollis", "Anterior neck flexion and antecollis (MDS consensus)", r"^post_E7_neck$"),
    ("chorea_generalized", "Generalized chorea (UHDRS maximal chorea)", r"^invol_chorea_E\d$"),
    ("chorea_region", "Chorea by region (UHDRS maximal chorea)", r"^invol_chorea_E1_(?:trunk|arms|legs)$"),
    ("chorea_fingers", "Choreic finger movements (piano-playing fingers)", r"^invol_chorea_E2_fingers$"),
    ("impersistence", "Motor impersistence", r"^invol_impersistence_"),
    ("hemiballism", "Hemiballism", r"^invol_hemiballism_"),
    ("athetosis", "Athetosis (DIS)", r"^invol_athetosis_"),
    ("choreoathetosis", "Choreoathetosis (DIS)", r"^invol_choreoathetosis_"),
    ("pseudoathetosis", "Pseudoathetosis (proprioceptive loss)", r"^invol_pseudoathetosis_"),
    ("hd_dystonia", "Dystonia in Huntington's disease (UHDRS maximal dystonia)", r"^invol_hd_dystonia_"),
    ("lid_peak", "Peak-dose levodopa-induced dyskinesia (UDysRS)", r"^invol_lid_peak_"),
    ("lid_dystonic", "Peak-dose dystonic levodopa-induced dyskinesia (UDysRS)", r"^invol_lid_dystonic_"),
    ("lid_diphasic", "Diphasic levodopa-induced dyskinesia (UDysRS)", r"^invol_lid_diphasic_"),
    ("tardive", "Tardive dyskinesia (AIMS; the orobuccolingual region is not shown)", r"^invol_td_(?!stereotypy)"),
    ("stereotypy", "Motor stereotypies", r"^invol_stereotypy_"),
    ("tardive_stereotypy", "Tardive stereotypies (AIMS)", r"^invol_td_stereotypy_"),
    ("akathisia", "Akathisia (BARS, objective rating)", r"^invol_akathisia_"),
    ("rls", "Restless legs syndrome", r"^invol_rls_"),
    ("moving_toes", "Painful legs and moving toes", r"^invol_moving_toes_"),
    ("mirror", "Mirror movements (Woods and Teuber)", r"^invol_mirror_"),
    ("cervical", "Cervical dystonia (TWSTRS)", r"^post_E7_cd_"),
    ("writers_cramp", "Writer's cramp (WCRS)", r"^post_W_(?:writing|writers_cramp)$"),
    ("task_specific", "Action dystonia (posture-induced)", r"^post_E2_task_specific_arm$"),
    ("foot_dystonia", "Foot dystonia", r"^post_E\d_(?:foot_dystonia|off_period_foot)$"),
    ("distribution", "Hemidystonia, segmental and multifocal dystonia",
     r"^post_E7_(?:hemidystonia|segmental|multifocal)$"),
    ("generalized", "Generalized dystonia (torsion dystonia)", r"^post_E7_torsion_dystonia$"),
    ("overflow", "Overflow dystonia", r"^post_W_overflow$"),
    ("mirror_dystonia", "Mirror dystonia", r"^post_W_(?:writing_both|mirror)$"),
    ("opisthotonus", "Opisthotonus", r"^post_E7_opisthotonus$"),
    ("fixed", "Fixed dystonia", r"^post_E1_fixed_"),
    ("pkd", "Paroxysmal kinesigenic dyskinesia", r"^post_E7_pkd$"),
    ("pnkd", "Paroxysmal non-kinesigenic dyskinesia", r"^post_E1_pnkd_"),
    ("ped", "Paroxysmal exercise-induced dyskinesia", r"^post_E7_ped_"),
    ("stance", "Stance (SARA item 2)", r"^atax_SARA2_"),
    ("romberg", "Romberg test (feet together, eyes open then closed)", r"^atax_ROMB"),
    ("sitting", "Sitting (SARA item 3)", r"^atax_SARA3_"),
    ("titubation", "Titubation", r"^atax_TITUB$"),
    ("ddk", "Dysdiadochokinesia (SARA item 7)", r"^atax_SARA7_"),
    ("heel_shin", "Heel-shin slide (SARA item 8)", r"^atax_SARA8_"),
    ("ref_level0", "Level 0 clips (neutral body)", r"^(?:jerk|invol)_E\d"),
]
SUBGROUP_RANK = {k: i for i, (k, _, _) in enumerate(SUBGROUPS)}


def subgroup_of(meta, notes=None):
    """(key, heading) of a clip's heading inside its group: an exam clip under its scale item (ITEM_NAMES, as on the
    exam page), a catalogue clip under its clinical entity (SUBGROUPS); ('other', 'Other') otherwise."""
    if not is_catalogue(meta):
        return "item:" + meta["item"], ITEM_NAMES.get(meta["item"], meta["item"])
    key = cat_series_key(meta["clip_id"])
    for k, lab, pat in SUBGROUPS:
        if re.search(pat, key):
            return k, lab
    if notes is not None:
        notes.append(f"heading: {key}")
    return "other", "Other"


def scale_name(item):
    """The scale item of a catalogue clip as its specification writes it; a bare MDS-UPDRS part III number (with or
    without the scale's name) takes its item title ('3.16 Kinetic tremor of the hands', '3.10 Gait (level 0)')."""
    m = re.fullmatch(r"(?:MDS-UPDRS )?(3\.\d+)(?: (.+))?", item or "")
    if m and m.group(1) in MDS_UPDRS_ITEMS:
        return f"{m.group(1)} {MDS_UPDRS_ITEMS[m.group(1)]}" + (f" ({m.group(2)})" if m.group(2) else "")
    return item


SCALE_NAMES = ("MDS-UPDRS", "TETRAS", "SARA", "ICARS", "UMRS", "YGTSS", "MRVS", "SCATS", "UHDRS", "UDysRS", "AIMS",
               "BARS", "TWSTRS", "BFMDRS", "WCRS", "DIS", "Conn", "FTM", "Woods and Teuber", "PSPRS", "S-FMDRS")
#   the scales the clips' texts name, in the order of the page's about section (the public gallery's groups, 2026-11-10)
SCALE_PATTERNS = {     # a scale's spellings in the texts, when not its name alone (case-sensitive)
    "MDS-UPDRS": r"\bMDS-UPDRS\b|^3\.\d+\b",               # the exam's items are bare MDS-UPDRS part III numbers
    "BFMDRS": r"\bBFM(?:DRS)?\b", "FTM": r"\bFTM\b|\bFahn-Tolosa-Marin\b", "S-FMDRS": r"(?<![\w-])S-FMDRS\b",
    "DIS": r"(?<![\w-])DIS\b"}


def scales_named(texts):
    """{scale names} of SCALE_NAMES that a clip's texts name (a string or an iterable of strings; an MDS-UPDRS part
    III item number counts at the start of a text)."""
    texts = [texts] if isinstance(texts, str) else list(texts)
    out = set()
    for s in SCALE_NAMES:
        pat = SCALE_PATTERNS.get(s, rf"(?<![\w-]){re.escape(s)}\b")
        if any(re.search(pat, t or "") for t in texts):
            out.add(s)
    return out


def overlays(meta):
    """[{kind, from, to, text}] of a clip's per-time texts (the data file's 'overlays': the Romberg eye state, the
    jerks' stimuli, the dual task), times in s on the clip's clock rounded to 1 ms, the text the truth's own; []
    without any."""
    return [{"kind": o["kind"], "from": round(float(o["from_s"]), 3), "to": round(float(o["to_s"]), 3),
             "text": str(o["text"])} for o in (meta.get("overlays") or [])]


def catalogue_texts(meta):
    """The texts of a catalogue clip (the keys of page_texts, and 'overlays' when the clip carries per-time texts)
    and the notes; the chip is set per series by series_chips."""
    notes = []
    key = cat_series_key(meta["clip_id"])
    if key in CAT_SERIES_NAMES:
        series = CAT_SERIES_NAMES[key]
    else:
        series = meta["operator"]
        notes.append(f"series: {key}")
    out = {"operator": meta["operator"], "grade": meta["grade_text"], "params": meta["params_text"] or "",
           "family": GROUP_NAMES[group_of(meta)], "item": scale_name(meta["item"]), "segment": segment_name(meta),
           "series": series, "series_key": key, "chip": ""}
    ov = overlays(meta)
    if ov:
        out["overlays"] = ov
    return out, notes


# ---------------------------------------------------------------- grade chips and list lines of a series
TEXT_GRADE_RANK = {"mild": 0, "moderate": 1, "severe": 2}
#   the order of the text grades inside a series (catalogue_rows): 'mild (described level)' before 'moderate ...'


def text_grade_rank(meta):
    """The rank of a clip's text grade inside its series (TEXT_GRADE_RANK on its first word), 0 otherwise."""
    g = meta.get("grade")
    return TEXT_GRADE_RANK.get(g.split(" ", 1)[0], 0) if isinstance(g, str) else 0


CHIP_KEY_NAMES = {"frequency": "frequency", "amplitude": "amplitude", "conn": "Conn",
                  "umrs_negative_severity": "severity", "stimulus_sensitivity": "stimulus sensitivity",
                  "ygtss_number": "number", "mrvs_frequency": "MRVS frequency", "ygtss_intensity": "intensity",
                  "ygtss_complexity": "complexity", "sfmdrs_severity": "severity", "sfmdrs_duration": "duration"}
CHIP_FOCUS = {"jerk_tics_E1_complexity": ("ygtss_complexity",), "jerk_tics_E7_complexity": ("ygtss_complexity",),
              "jerk_tics_E1_bouts_60s": ("ygtss_number", "ygtss_intensity"),
              "jerk_tics_E1_touch": ("ygtss_complexity",), "jerk_tics_E7_touch": ("ygtss_complexity",)}
CHIP_STR = {"compensated": "compensated (Duchenne)", "fixed_fist": "clenched fist", "fixed_foot": "inverted foot",
            "episode": "attack", "excerpt": "excerpt of an attack"}
#   a text grade's chip: the grade as the specification writes it, but for these
CHIP_ID = {"tremor_E7_orthostatic_g0": "level 0", "tremor_E7_orthostatic_present": "present",
           "gait_E8_functional_knee_buckling": "knee buckling", "gait_E8_functional_astasia_abasia": "astasia-abasia",
           "gait_E8_functional_slow_motion": "slow motion", "gait_E8_functional_walking_on_ice": "walking on ice",
           "gait_E8_functional_dragging_left": "dragging leg",
           "gait_E8_functional_fluctuating": "momentary fluctuations",
           "post_E7_lumbar_mild": "mild anterior flexion", "post_E7_lumbar_open": "camptocormia",
           "post_E7_thoracic_mild": "mild anterior flexion", "post_E7_lateral_mild": "mild lateral flexion",
           "post_E7_lateral_open": "Pisa syndrome", "post_E7_neck_mild": "mild anterior flexion",
           "post_E7_neck_open": "antecollis", "post_E7_cd_shift_lateral": "present",
           "post_E7_cd_shift_anterior": "present", "post_E7_cd_shift_posterior": "present",
           "post_E7_pkd_dystonic": "dystonic", "post_E7_pkd_choreic": "choreic",
           "park_E4aL_ulfreeze_long": "repeated freezing (no grade)", "atax_ROMB_sensory_bedside": "positive sign",
           "isometric_tremor_E4b_fist_g0": "level 0", "isometric_tremor_E4b_fist_present": "present",
           "orthostatic_myoclonus_E7_present": "present", "park_E7_arise_abrupt": "abrupt arising",
           "park_E22_sitdown_enbloc": "en bloc", "park_E1_motorstate_offon": "OFF to ON",
           "park_E1_motorstate_onoff": "ON to OFF (wearing-off)", "park_E7_gowers": "present",
           "gait_E8_vestibular_left_eyes_open": "eyes open", "gait_E19_vestibular_left_eyes_closed": "eyes closed",
           "gait_E8_sensory_ataxic_eyes_open": "eyes open", "gait_E19_sensory_ataxic_eyes_closed": "eyes closed"}
CHIP_QUAL = {   # two clips of one grade in a series: the grade number and what tells them apart
    "invol_chorea_E1_g2": "mild, common", "invol_chorea_E1_g2b": "moderate, intermittent",
    "invol_chorea_E7_g2": "mild, common", "invol_chorea_E7_g2b": "moderate, intermittent",
    "invol_mirror_E4a_g2b": "stronger but briefer",
    "park_E7_arise_g1_slow": "slow", "park_E7_arise_g1_attempts": "more than one attempt",
    "post_E7_cd_shoulder_D1_mild_intermittent": "mild, intermittent",
    "post_E7_cd_shoulder_D2_moderate_constant": "moderate, constant",
    "post_E7_cd_shoulder_D2_severe_intermittent": "severe, intermittent"}
CHIP_TEXT = [   # (regex on the grade text, chip): a text grade read from its grade text, numbers as written there
    (r"^DIS amplitude (\d+), duration (\d+)$", r"amplitude \1, duration \2"),
    (r"posturographic Romberg ratio (\d+(?:\.\d+)?) \(published means\)$", r"ratio \1")]
LEVEL_SERIES = ("park_E4aL_sequence_ft", "park_E4L_sequence_ps")   # numeric grades that are levels, not grades
CHIP_PREFIX_SERIES = {"post_E7_torsion_dystonia": "severity",    # a numeric grade that is a scale's factor (BFMDRS)
                      "park_E21_applause": "score"}
SERIES_SEE_ALSO = {   # a catalogue series whose missing grade is an exam clip under another heading (named in the
    #                     series' list line)
    "post_E7_cd_rotation_duration": "dystonia_E7_neck_rotation_left_g2_duration3"}
_NUMERIC_CHIP = re.compile(r"^\d+(?:\.\d+)?(?: \(.*\))?$")
_LEVEL_CHIP = re.compile(r"^level (\d+)$")
_LABEL_CHIP = re.compile(r"(\D+?) (\d+(?:\.\d+)?)")                 # 'duration 4', 'severity 2', 'ratio 1.5'


def _chip_pairs(chip):
    """[(label, integer)] of a chip made of 'label n' parts joined by ', ' ('amplitude 2, duration 3'), else None."""
    out = []
    for part in chip.split(", "):
        m = re.fullmatch(r"(\D+?) (\d+)", part)
        if not m:
            return None
        out.append((m.group(1), int(m.group(2))))
    return out


def _lead_num(g):
    """The number a grade starts with ('2b' gives 2)."""
    m = re.match(r"\d+(?:\.\d+)?", str(g))
    return m.group(0) if m else str(g)


def _text_chip(m):
    """The chip of a text grade: CHIP_STR, else CHIP_TEXT on its grade text, else the grade as written."""
    g = m["grade"]
    if g in CHIP_STR:
        return CHIP_STR[g]
    for pat, rep in CHIP_TEXT:
        mm = re.search(pat, m.get("grade_text") or "")
        if mm:
            return mm.expand(rep)
    return g.replace("_", " ")


def _is_num(g):
    return isinstance(g, (int, float)) and not isinstance(g, bool)


def _fmt_num(g):
    return f"{g:g}" if isinstance(g, float) else str(g)


def series_chips(ms, notes=None):
    """{clip: chip} of the clips of one catalogue series (their metas, in order): a numeric grade (with the exam's
    a/b mark of grade 1); a dict grade by its keys that vary in the series (all keys, or CHIP_FOCUS, when none
    varies); a text grade through _text_chip; a level 0 'level 0' unless its series is numeric; a posture without its
    jerks or tremor 'posture only'; the grade number and CHIP_QUAL when two clips share a grade; 'level n' in
    LEVEL_SERIES; CHIP_ID first."""
    dicts = [m["grade"] for m in ms if isinstance(m["grade"], dict)]
    keys = []
    for g in dicts:
        keys += [k for k in g if k not in keys]
    vary = [k for k in keys if len({repr(g.get(k)) for g in dicts}) > 1]
    numeric = any(_is_num(m["grade"]) and m["level0"] is not None for m in ms)
    out = {}
    for m in ms:
        cid, g = m["clip_id"], m["grade"]
        key = cat_series_key(cid)
        if cid in CHIP_ID:
            chip = CHIP_ID[cid]
        elif cid in CHIP_QUAL:
            chip = f"{_lead_num(g)} ({CHIP_QUAL[cid]})"
        elif key in LEVEL_SERIES and _is_num(g):
            chip = f"level {_fmt_num(g)}"
        elif key in CHIP_PREFIX_SERIES and _is_num(g):
            chip = f"{CHIP_PREFIX_SERIES[key]} {_fmt_num(g)}"
        elif re.search(r"_posture(?:_only)?$", cid):
            chip = "posture only"
        elif isinstance(g, dict):
            ks = vary or list(CHIP_FOCUS.get(key, ())) or list(g)
            parts = []
            for k in ks:
                if k in g:
                    if k not in CHIP_KEY_NAMES and notes is not None:
                        notes.append(f"chip key: {k}")
                    parts.append(f"{CHIP_KEY_NAMES.get(k, k.replace('_', ' '))} {_fmt_num(g[k])}")
            chip = ", ".join(parts)
        elif isinstance(g, str):
            chip = _text_chip(m)
        elif m["level0"] is None and not numeric:
            chip = "level 0" if g == 0 else _fmt_num(g)
        else:
            sub = re.search(r"_g\d+([ab])(?=_|$)", cid)
            chip = _fmt_num(g) + (FLOOR_MARKS[sub.group(1)] if sub else "")
        out[cid] = chip
    return out


def series_summary(ms, chips):
    """The line under a series in the list: 'grades 0, 1, 2' for a numeric series, 'levels 0, 1, 2, 3' for a level
    ladder, one label once ('duration 1, 2, 4, 5'), the range of every varying key of a dict-graded series
    ('amplitude 1 to 3, duration 2 to 4'), else its text chips then its numeric ones after 'grade' or 'grades'."""
    labs = [re.sub(r" \(.*\)", "", chips[m["clip_id"]]) for m in ms]
    uniq = []
    for x in labs:
        if x not in uniq:
            uniq.append(x)
    if len(uniq) > 1 and all(_LEVEL_CHIP.match(x) for x in uniq):
        return "levels " + ", ".join(_LEVEL_CHIP.match(x).group(1) for x in uniq)
    one = [_LABEL_CHIP.fullmatch(x) for x in uniq]
    if all(one) and len({x.group(1) for x in one}) == 1:
        return f"{one[0].group(1)} " + ", ".join(x.group(2) for x in one)
    pairs = [_chip_pairs(chips[m["clip_id"]]) for m in ms]
    same_keys = all(p and len(p) > 1 and [k for k, _ in p] == [k for k, _ in pairs[0]] for p in pairs)
    if len(ms) > 1 and not any(isinstance(m["grade"], dict) for m in ms) and same_keys:
        out = []
        for i, (k, _) in enumerate(pairs[0]):
            vals = sorted({p[i][1] for p in pairs})
            out.append(f"{k} {vals[0]}" if len(vals) == 1 else f"{k} {vals[0]} to {vals[-1]}")
        return ", ".join(out)
    if all(_NUMERIC_CHIP.match(chips[m["clip_id"]]) for m in ms):
        return ("grades " if len(uniq) > 1 else "grade ") + ", ".join(uniq)
    dict_ms = [m for m in ms if isinstance(m["grade"], dict)]
    if dict_ms:
        head = [chips[m["clip_id"]] for m in ms if not isinstance(m["grade"], dict)]
        keys = []
        for m in dict_ms:
            keys += [k for k in m["grade"] if k not in keys]
        vary = [k for k in keys if len({repr(m["grade"].get(k)) for m in dict_ms}) > 1]
        ks = vary or list(CHIP_FOCUS.get(cat_series_key(dict_ms[0]["clip_id"]), ())) or keys
        parts = []
        for k in ks:
            vals = sorted({m["grade"][k] for m in dict_ms if k in m["grade"]})
            lab = CHIP_KEY_NAMES.get(k, k.replace("_", " "))
            parts.append(f"{lab} {_fmt_num(vals[0])}" if len(vals) == 1
                         else f"{lab} {_fmt_num(vals[0])} to {_fmt_num(vals[-1])}")
        return ", ".join(head + parts)
    words = [x for x in uniq if not _NUMERIC_CHIP.match(x)]          # 'posture only, grades 2, 3'
    wsep = "; " if any("," in x for x in words) else ", "
    nums = [x for x in uniq if _NUMERIC_CHIP.match(x)]
    if nums:
        words.append(("grades " if len(nums) > 1 else "grade ") + ", ".join(nums))
    return wsep.join(words)


def catalogue_title(series_label, heading):
    """The page title of a catalogue series: 'heading: label', the heading alone when the two say the same."""
    if series_label.strip().lower() == heading.strip().lower():
        return heading
    return f"{heading}: {series_label}"


# ================================================================ wave SH1d: the clinical caption and its provenance
# The clinical review of the English page (2026-11-03, PLAN.txt of the session, minors carried to lane 1): the marks of
# provenance in the specifications' texts ('[module]', '(module level)', '(D4)', 'declared map', 'estimator floor' ...)
# mean nothing to a clinician in a caption. The catalogue page shows each text without them (split_provenance) and
# lists them in a provenance line under the caption, each with the phrase it qualifies and what it means. The
# specifications' texts are not rewritten: the split is a reading of them, and a text without a mark passes unchanged.
PROVENANCE_KINDS = {   # kind: what the mark means, after the page's about section (task SKELV4b)
    "module": "a value the generator chose, with no published source",
    "declared_map": "the generator's own reading of the grade as a movement, with no published mapping",
    "declared_ladder": "a grading of the generator's own, with no published scale",
    "cited": "a value from the clip's cited sources",
    "rule_d4": "read at the wrist, by the program's reading rule for a forearm tremor",
    "estimator_above": "above the detection floor of the program's amplitude estimator",
    "estimator_below": "below the detection floor of the program's amplitude estimator",
    "draft": "one of the starting postures of the program's video draft",
}
PROVENANCE_PARTS = [   # (regex on a whole part of a bracketed group, kind): the part is a mark of provenance
    (r"module(?: [a-z' ]+)?", "module"),
    (r"D4", "rule_d4"), (r"declared ladder", "declared_ladder"), (r"declared(?: map)?", "declared_map"),
    (r"cited(?: [a-z]+)?", "cited"),
    (r"above the estimator floor", "estimator_above"), (r"below the estimator floor", "estimator_below"),
    (r"video draft starting posture", "draft")]
GRADE_ABOUT = "grade"   # a declared map or ladder and an estimator-floor mark qualify the grade as a whole, wherever
#   their mark stands
GRADE_KINDS = ("declared_map", "declared_ladder", "estimator_above", "estimator_below")
CARRIED_KINDS = ("module", "cited")    # a mark of these kinds with words after its first ('module item scores',
#   'cited phases') names what of the phrase it qualifies: the words follow the phrase in brackets ('WCRS (item
#   scores)'), so that the provenance line does not read as if the scale itself had no source (wave SH1d review of
#   lane page) and the phrase stays grammatical ('rising over the last 10 cm (envelope)'), but those of
#   PLAIN_MARK_WORDS, which only restate the mark. 'on the level 0's clock' is no mark: it says how a duration is
#   timed and stays in the caption (wave SH1d, review r3 of lane page)
PLAIN_MARK_WORDS = ("docstring", "comment above", "value", "values", "reading", "default", "level", "normal")
_PROV_PART = [(re.compile(rf"^{p}$"), k) for p, k in PROVENANCE_PARTS]
_GROUP = re.compile(r"\s?([\[(])([^\[\]()]*)([\])])")               # an innermost bracketed group
_LOOSE = re.compile(r"(,\s)?\bdeclared (map|ladder)\b(\s)?")         # 'declared map' outside a bracket
_LOOSE_CITED = re.compile(r"\bthe cited normal rate\b")             # 'cited' inside a phrase (the tapping level 0s)
LOOSE_CITED_ABOUT = "the normal rate"


def _prov_kind(part):
    """The kind of a part of a bracketed group when it is a mark of provenance (PROVENANCE_PARTS), else None."""
    for rx, kind in _PROV_PART:
        if rx.match(part.strip()):
            return kind
    return None


def _about(about, part, kind):
    """The phrase a mark qualifies: about, with the mark's words after its first in brackets for CARRIED_KINDS
    ('WCRS (item scores)')."""
    rest = part.strip().split(" ", 1)[1:]
    if kind in CARRIED_KINDS and rest and rest[0] not in PLAIN_MARK_WORDS and about != GRADE_ABOUT:
        return f"{about} ({rest[0]})" if about else rest[0]
    return about


def _clause_before(text, end, stops=",;:"):
    """The clause of text that ends at end: back to the previous stop character outside brackets (',', ';' or ':'
    by default), or to an opening bracket that encloses it, or to the start; stripped."""
    depth, i = 0, end
    while i > 0:
        ch = text[i - 1]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                break
            depth -= 1
        elif ch in stops and depth == 0:
            break
        i -= 1
    return text[i:end].strip()


def _tidy(text):
    """Spaces and punctuation left by a removed mark: no space before ',', ';', ':' or a closing bracket, one space,
    no empty bracket, no doubled delimiter."""
    t = re.sub(r"\s+([,;:)\]])", r"\1", text)
    t = re.sub(r"[(\[]\s*[)\]]", "", t)
    t = re.sub(r"([,;])\s*([,;])", r"\2", t)
    t = re.sub(r"\s{2,}", " ", t)
    return re.sub(r"^[\s,;:]+|[\s,;]+$", "", t).strip()


def split_provenance(text):
    """(clinical text, notes) of a specification's text: every bracketed group made only of marks of provenance
    (PROVENANCE_PARTS, its parts split on ', ' and '; ') is removed, a group that mixes them keeps its other parts,
    innermost groups first ('(1.5 x the 10 deg cut-off [module])' keeps '(1.5 x the 10 deg cut-off)'), and 'declared
    map' or 'declared ladder' outside a bracket is removed with its comma, 'the cited normal rate' reads 'the normal
    rate' (its note first, LOOSE_CITED_ABOUT). notes: [{'about', 'mark': the mark as
    written, 'kind'}] in reading order; 'about' is the phrase the mark qualifies: the clause before a mark in round
    brackets (back to ',', ';' or ':', the specifications' style '5 Hz (module)'), the clause back to ';' or ':'
    before a mark in square brackets (the postures' style, a mark closing a description), GRADE_ABOUT for a
    mark of GRADE_KINDS, with the mark's own words after its first for CARRIED_KINDS ('WCRS (module item scores)':
    'WCRS (item scores)', _about). A text without a mark is returned unchanged with no note."""
    t, notes = str(text or ""), []
    for _ in _LOOSE_CITED.finditer(t):                  # first: in the specifications it precedes the groups
        notes.append({"about": LOOSE_CITED_ABOUT, "mark": "cited", "kind": "cited"})
    t = _LOOSE_CITED.sub(LOOSE_CITED_ABOUT, t)
    while True:
        changed = False
        for m in _GROUP.finditer(t):
            parts = re.split(r"(?:,|;)\s", m.group(2))
            kinds = [_prov_kind(p) for p in parts]
            if not any(kinds):
                continue
            about = _clause_before(t, m.start(), ",;:" if m.group(1) == "(" else ";:")
            for p, k in zip(parts, kinds):
                if k:
                    notes.append({"about": GRADE_ABOUT if k in GRADE_KINDS else _about(about, p, k),
                                  "mark": p.strip(), "kind": k})
            keep = [p for p, k in zip(parts, kinds) if not k]
            new = f" {m.group(1)}{', '.join(keep)}{m.group(3)}" if keep else ""
            t = t[:m.start()] + new + t[m.end():]
            changed = True
            break
        if not changed:
            break
    while True:
        m = _LOOSE.search(t)
        if not m:
            break
        notes.append({"about": GRADE_ABOUT, "mark": m.group(0).strip(" ,"), "kind": f"declared_{m.group(2)}"})
        t = t[:m.start()] + (" " if m.group(1) and m.group(3) else "") + t[m.end():]
    if not notes:
        return str(text or ""), []
    return _tidy(t), notes


CLINICAL_FLOOR_MARKS = {" (above the estimator floor)": " (larger amplitude)",
                        " (below the estimator floor)": " (smaller amplitude)"}
#   the grade chips of the two grade 1 clips (ids _g1a, _g1b) on the catalogue page: what tells them apart for a
#   clinician (both inside the grade 1 band); the estimator floor goes to the provenance line (split_provenance)


def clinical_chip(chip):
    """A grade chip of the catalogue page: the estimator-floor marks of CLINICAL_FLOOR_MARKS in clinical words."""
    for old, new in CLINICAL_FLOOR_MARKS.items():
        chip = chip.replace(old, new)
    return chip


def clinical_texts(text):
    """(text, provenance) of a clip's page texts on the catalogue page: the operator, grade, parameter and scale
    item texts without their marks of provenance (split_provenance; the item is the scale in the breadcrumb), the
    chip in clinical words (clinical_chip), and the provenance line: [{'about', 'meaning', 'marks'}] in reading order,
    one entry per qualified phrase (its marks' meanings joined, PROVENANCE_KINDS), [] when no text carries a mark."""
    out, notes = dict(text), []
    for k in ("operator", "grade", "params", "item"):
        out[k], n = split_provenance(text.get(k) or "")
        notes += n
    out["chip"] = clinical_chip(text.get("chip") or "")
    lines, index = [], {}
    for n in notes:
        if n["about"] not in index:
            index[n["about"]] = len(lines)
            lines.append({"about": n["about"], "meaning": [], "marks": []})
        e = lines[index[n["about"]]]
        mean = PROVENANCE_KINDS[n["kind"]]
        if mean not in e["meaning"]:
            e["meaning"].append(mean)
        e["marks"].append(n["mark"])
    return out, [{"about": e["about"], "meaning": "; ".join(e["meaning"]), "marks": e["marks"]} for e in lines]


# ---------------------------------------------------------------- level 0 clips shown without a symptom clip
STANDALONE_LEVEL0 = {   # a catalogue level 0 kept on the page although no clip names it as its level 0 (the page's
    #                     rule, render_skeleton_catalogue.hidden_clips, hides such a level 0 otherwise)
    "atax_SARA8_heelshin_right_g0": "the right side of the bilateral heel-shin test, built as its own reference "
                                    "(shared wave SH7, with the couch views of the right leg)"}
