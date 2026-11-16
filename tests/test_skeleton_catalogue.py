"""Tests of the skeleton page of the symptom catalogue (tasks SKELV2 to SKELV4b): scripts/render_skeleton_catalogue.py
--set catalogue and the structure of its list, scripts/skeleton_text.py.

Owner decision of 2026-11-03: the project is published, so the page is in English (it was French until then). The
operator, grade and parameter texts of a clip are its specification's own and the per-time texts the truth's own;
skeleton_text holds only the structure (groups, headings, series, chips). The pinned digests of the exam page
changed with that decision.

- --set exam: the page template and the exam tables are pinned (digests), the defaults are the exam set,
  skeleton_v1 and its run record, and the exam page carries no catalogue patch;
- every catalogue clip of the seven modules (catalogue.CATALOGUE_MODULES, the default --modules) shows its
  specification's own texts, gets a group, a heading, a series label and a grade chip, with no note and no French
  left (no accented letter, no French word) in any text of the page;
- the list: the groups in the clinical order of the task, every series inside one group, distinct chips inside a
  series, no series key shared by an exam clip and a catalogue clip;
- refusal: a clip whose own module validity fails (the gait generator's 'gait': {'ok': False}) is refused in the
  catalogue set and kept in the exam set; refused clips are listed in the manifest and on the page, not embedded;
- page_texts() never raises: a series without a label shows its clip's operator and is listed ('unlabelled' in the
  manifest); the page is not refused for it;
- provenance: a catalogue data file carries the digests of its module and of the mdfm files that module imports,
  and a file whose digests differ from the current files' is stale (rebuilt by a rerun, refused by the page); the
  families left out of --modules are never imported;
- the catalogue page: no series id in the list filter, a dystonic-posture level 0 not tagged 'Normal', the refused
  clips named, the identifiers explained;
- task SKELV3: the per-time texts of a clip's truth (the Romberg eye state, the jerks' stimuli) are exported with
  the clip and drawn by the page in the truth's words; the scene boxes come from the scene mesh's parts (a non-box
  part is left out and listed); the exam page rebuilt from the skeleton_v1 data is pinned;
- task SKELV4a (the shared wave): every view of viewer.VIEWS reaches the catalogue page under the viewer's own title
  while the exam page keeps the views and regions of task SKELV; the page's own views do not repeat the viewer's;
  every view's region frames at least two points (the toes with their landmarks); an oriented box of the scene (the
  examiner's finger) is written as its 8 corners and drawn by the 12 edges of viewer.box; a supine clip on the couch
  takes the room floor and the couch's heading of ataxia.bed_view, opens on the couch views and never follows the
  pelvis; the SARA 8 clips, the touching tics and the scissoring gait (mild before moderate) under their headings;
  the gaits' leg-to-leg check is reported, not a refusal;
- --data-from: a preview page built from another folder's data, nothing written there;
- wave SH1d (the clinical review of the English page): a clip drawn on its level 0's floor and chair (the page reads
  them from the index), a level 0's floor over its standing frames; the uniform checks of the skeleton (continuity,
  self-contact, skate, statue, the floor of the pair) each flag a synthetic offender and pass its clean twin, and
  since 2026-11-11 refuse (the page hides the clips they flag and the clips drawn beside them); the skate check reads
  the sole cloud on its touching frames, so that a foot pivoting about a sole point passes and a foot gliding, or
  slipping at its pivot, fails, its tolerance growing with the turn by the cloud's covering radius; the marks of
  provenance moved out of the clinical caption into a provenance line; the TETRAS heading of the simple kinetic
  tremor; a level 0 no clip names hidden, but the standalone ones.
Specs only, or procedural bodies (no AMASS read, no lockbox, EVAL, CODY or care_pd record); files go to pytest's
temporary folder only.
"""
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import os
import re

import pytest

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_body = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")

# The exam page: sha256 of PAGE_TEMPLATE and of the exam tables (json of FAMILY_NAMES, FAMILY_ORDER, ITEM_NAMES,
# SERIES_NAMES, EXAM_VIEWS). They pinned the French page of task SKELV (commit b1638d8) until the owner's decision of
# 2026-11-03 (everything for a reader in English); these are the English page's (after its review: the official
# MDS-UPDRS titles of 3.6 and 3.14, the estimator floor named in the grade 1 chips, the title capitalized, 'symptom'
# for 'operator' in the about section); since the open-body port's wave 7 the about section names the Anny body model
# and the procedural references (5e19c340 before)
SKELV_TEMPLATE_SHA256 = "35959d858724987e635c8bdeab2bed4e1aab52b9c3f158a5760a4ff0b0eb4f2d"
# wave SH1d: the tables' digest with the series label of the raise level 0 with the longer rest; wave 7 of the
# open-body port: the references' labels without AMASS (38c93e51 before)
SKELV_TABLES_SHA256 = "eb7205c31c9b81869d1890c1121fdd24cef63b09d3146c8fb6c67a09a8e344d7"
EXAM_TABLES = ("FAMILY_NAMES", "FAMILY_ORDER", "ITEM_NAMES", "SERIES_NAMES", "EXAM_VIEWS")
TASK_GROUPS = ["Tremors", "Finger-to-nose test (dysmetria)", "Bradykinesia", "Parkinsonism, other signs",
               "Atypical parkinsonism (PSP)", "Parkinsonian postures (camptocormia, Pisa syndrome, antecollis)",
               "Chorea, ballism, athetosis", "Dyskinesias (levodopa-induced, tardive)",
               "Stereotypies, akathisia, restless legs", "Mirror movements", "Myoclonus",
               "Negative myoclonus (asterixis)", "Tics", "Startle", "Clonus", "Dystonia", "Paroxysmal dyskinesias",
               "Ataxia", "Freezing of gait", "Gait", "Healthy references"]
EXAM_PAGE = "<store>/synth_viewer/skeleton_v1"          # the SKELV exam data (outside the tree)
# the English exam page built from them (owner's decision of 2026-11-03; the French page of task SKELV had e7495a62;
# 7257a164 until the open-body port's wave 7 changed the template's about section)
EXAM_PAGE_SHA256 = "4e0c04bb9803c9627d835e474007bf4f629cf16f14891177258fbdf4039bf07a"
# French left in a text: an accented letter, the French quotes, or a common French word ('et al.' and the English
# clinical loans 'en bloc' and 'geste antagoniste' are English)
FRENCH = re.compile(r"[\u00c0-\u00ff\u0152\u0153\u00ab\u00bb]|\b(?:le|la|les|des|du|aux|avec|sans|pour|une|dans|sur"
                    r"|niveaux?|gauche|droite?|jambes?|bras|tremblements?|marche|assise?|debout|yeux|secousses?"
                    r"|squelette|symptômes?|vue|clips? dont|identifiant|refusés?|grade de)\b|\bet\b(?! al\b)", re.I)


def _script():
    spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_script_v2",
                                                  os.path.join(paths.TREE, "scripts", "render_skeleton_catalogue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S():
    return _script()


@pytest.fixture(scope="module")
def cat_specs(S):
    from mdfm.synth import catalogue as CAT
    return CAT.catalogue_set(modules=S.DEFAULT_MODULES)


def spec_meta(s, params_text=None):
    """The metadata fields page_texts and catalogue_rows read, from a spec (the parameter text of exam.describe
    when given)."""
    from mdfm.synth import exam as EX
    return {"clip_id": s.clip_id, "family": s.family, "segment": s.segment, "segment_title": EX.segment_title(s),
            "item": s.item, "operator": s.operator, "grade": s.grade, "grade_text": s.grade_text,
            "params_text": s.params_text if params_text is None else params_text, "level0": s.level0,
            "since": s.since}


def french_left(text):
    """The French found in a text (an empty list when none)."""
    return [m.group(0) for m in FRENCH.finditer(text or "")]


def _page_text(html):
    """The page's own text and code outside its data blocks and its index (the clips' texts are checked apart),
    without the style sheet and the font names ('sans-serif')."""
    t = re.sub(r'<script type="text/plain" class="clipdata"[^>]*>[^<]*</script>', "", html)
    t = re.sub(r'<script type="application/json" id="index">.*?</script>', "", t, flags=re.S)
    t = re.sub(r"<style>.*?</style>", "", t, flags=re.S)
    return t.replace("sans-serif", "")


def _index(html):
    return json.loads(re.search(r'<script type="application/json" id="index">(.*?)</script>', html, re.S).group(1)
                      .replace("<\\/", "</"))


# ---------------------------------------------------------------- --set exam
def test_exam_set_defaults(S, capsys):
    assert hashlib.sha256(S.PAGE_TEMPLATE.encode("utf-8")).hexdigest() == SKELV_TEMPLATE_SHA256
    tables = json.dumps([getattr(S, k) for k in EXAM_TABLES], ensure_ascii=False)
    assert hashlib.sha256(tables.encode("utf-8")).hexdigest() == SKELV_TABLES_SHA256
    assert S.SET == "skeleton_v1" and S.OUT.endswith("/skeleton_v1") and S.RECORD.endswith("skeleton_v1.json")
    assert S.record_path(S.SET) == S.RECORD and S.SET_CATALOGUE == "skeleton_v2"
    from mdfm.synth import exam as EX
    exam_ids = [s.clip_id for s in EX.exam_set()]
    assert [s.clip_id for s in S.set_specs("exam", S.DEFAULT_MODULES)] == exam_ids
    S.main(["--list"])
    assert capsys.readouterr().out.strip().endswith(f"{len(exam_ids)} clips")
    S.main(["--set", "catalogue", "--list"])
    assert capsys.readouterr().out.strip().endswith(f"{len(S.set_specs('catalogue', S.DEFAULT_MODULES))} clips")
    assert '<html lang="en">' in S.PAGE_TEMPLATE and not french_left(_page_text(S.PAGE_TEMPLATE))


@needs_body
def test_exam_page_has_no_catalogue_patch(S, tmp_path):
    from mdfm.synth import exam as EX
    specs = {s.clip_id: s for s in EX.exam_set()}
    out = str(tmp_path / "skeleton_exam").replace("\\", "/")
    sel = [specs["tremor_E1_rest_g0"], specs["tremor_E1_rest_g2"]]
    for s in sel:
        assert S.export_clip(s, out)[1] is None
    metas, blobs = S.load_all(out, sel)
    page, _, _, notes = S.build_page(out, metas, blobs, [])
    with open(page, encoding="utf-8") as f:
        html = f.read()
    assert notes == {} and "<title>SYNTHETIC Skeleton catalogue</title>" in html
    assert "s.summary" not in html and "Refused clips" not in html and "c.en." not in html
    assert 'id="tagB"' not in html and "Identifiers:" not in html
    assert "fitBands" not in html and "drawEars" not in html and "landmark_bones" not in html   # SKELV3 review
    with gzip.open(S.data_path(out, "tremor_E1_rest_g0"), "rb") as f:
        assert "code" not in json.loads(f.read().decode("utf-8"))          # the exam data stay those of SKELV
    idx = _index(html)
    assert idx["set"] == "skeleton_v1" and idx["family_order"] == list(S.FAMILY_ORDER)
    assert idx["families"] == S.FAMILY_NAMES and all("en" not in c and "fr" not in c for c in idx["clips"])
    c = {x["id"]: x for x in idx["clips"]}["tremor_E1_rest_g2"]
    assert c["text"]["operator"] == specs["tremor_E1_rest_g2"].operator and c["text"]["chip"] == "2"
    assert c["text"]["grade"] == specs["tremor_E1_rest_g2"].grade_text
    assert c["text"]["segment"] == EX.SEGMENTS["E1"] and c["text"]["item"] == "3.17 Rest tremor amplitude"
    with gzip.open(S.data_path(out, "tremor_E1_rest_g2"), "rb") as f:
        d = json.loads(f.read().decode("utf-8"))
    assert d["catalogue"] == "skeleton_v1" and d["segment_title"] == EX.SEGMENTS["E1"]
    assert not french_left(_page_text(html)) and not french_left(json.dumps(idx["clips"], ensure_ascii=False))


# ---------------------------------------------------------------- the texts of every catalogue clip
def test_every_catalogue_clip_shows_its_own_texts(S, cat_specs):
    """The operator, grade and parameter texts are the specification's own; every clip gets a group, a heading, a
    series label and a chip (no note), and no text of the list holds French."""
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import exam as EX
    cat = [s for s in cat_specs if s.since == "catalogue_v1"]
    assert len(cat) > 300
    assert tuple(S.DEFAULT_MODULES) == CAT.CATALOGUE_MODULES and S.blocked_modules(S.DEFAULT_MODULES) == []
    fams = set()
    for s in cat:
        text, notes = S.page_texts(spec_meta(s))
        assert not notes, (s.clip_id, notes)
        assert (text["operator"], text["grade"], text["params"]) == (s.operator, s.grade_text, s.params_text or "")
        assert text["segment"] == EX.segment_title(s)
        for k in ("operator", "grade", "family", "item", "segment", "series"):
            assert text[k] and not french_left(text[k]), (s.clip_id, k, text[k])
        assert "\u2014" not in json.dumps(text) and "\u2013" not in json.dumps(text)
        fams.add(s.family)
    assert {"tremor", "reach", "myoclonus", "negative_myoclonus", "tics", "startle", "clonus", "gait", "chorea",
            "ballism", "athetosis", "dyskinesia", "stereotypy", "akathisia", "restless_legs", "moving_toes", "mirror",
            "parkinsonian_posture", "cervical_dystonia", "limb_dystonia", "dystonia_distribution", "paroxysmal",
            "parkinsonism", "ataxia"} <= fams
    modules = {getattr(s.build, "func", s.build).__module__.rsplit(".", 1)[-1] for s in cat}
    assert modules == set(CAT.CATALOGUE_MODULES)                     # every clip of the seven modules is checked
    rows, notes = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    assert notes == {}
    for m, text, g, sub in rows:
        for k in ("operator", "grade", "params", "family", "item", "segment", "series", "chip", "title", "scale"):
            assert not french_left(text.get(k)), (m["clip_id"], k, text.get(k))
        if m["since"] == "catalogue_v1":
            assert text["chip"] and text["summary"] and text["title"] and sub != "other", m["clip_id"]
            assert not french_left(text["summary"]), (m["clip_id"], text["summary"])


def test_no_french_left_on_the_page(S, cat_specs):
    """The catalogue template, its scripts and every table of skeleton_text are in English."""
    t = S.catalogue_template({"a_g1": ["gait"]}, {"a_g1": "clip a"}, {"b_g0": ["mdfm.synth.ataxia"]}, ["E1", "J2"])
    assert '<html lang="en">' in t and not french_left(_page_text(t)), french_left(_page_text(t))
    assert not french_left(_page_text(S.OVERLAY_JS + S.CATALOGUE_JS))
    TX = S.TX
    for table in (TX.GROUP_NAMES, TX.CAT_SERIES_NAMES, TX.ITEM_NAMES, TX.FAMILY_NAMES, TX.CHIP_ID, TX.CHIP_QUAL,
                  TX.CHIP_STR, TX.CHIP_KEY_NAMES, S.MODULE_NAMES):
        for k, v in table.items():
            assert not french_left(v), (k, v)
    for k, lab, _ in TX.SUBGROUPS:
        assert not french_left(lab), (k, lab)
    for _, lab in TX.SERIES_NAMES:
        assert not french_left(lab), lab
    for v in list(S.V.VIEWS.values()) + list(S.CATALOGUE_VIEWS.values()):
        assert not french_left(v.title), v.title


@needs_body
def test_described_parameter_texts_shown_as_they_are(S, cat_specs):
    """The clips whose caption text comes from exam.describe (a dystonic posture's truth) show that text."""
    from mdfm.synth import exam as EX
    by = {s.clip_id: s for s in cat_specs}
    for cid in ("jerk_mdyst_E7_neck_posture_only", "jerk_mdyst_E1_right_arm_posture_only",
                "tremor_E1_dystonic_head_posture"):
        s = by[cid]
        _, truth = s.build()
        described = EX.describe(s, truth)
        text, notes = S.page_texts(spec_meta(s, described))
        assert not notes and text["params"] == described and not french_left(described), (cid, described)


def test_numbers_keep_their_point(S, cat_specs):
    """Decimal grades and their list lines keep the decimal point; the page's clock and sizes too."""
    rows, _ = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    chip = {m["clip_id"]: text["chip"] for m, text, _, _ in rows}
    summ = {m["clip_id"]: text.get("summary") for m, text, _, _ in rows}
    assert chip["tremor_E2_essential_tetras2p5"] == "2.5" and chip["atax_ROMB_cerebellar"] == "ratio 1.5"
    assert summ["tremor_E2_essential_tetras2p5"] == "grades 1.5, 2, 2.5, 3, 3.5"
    assert summ["tremor_E2_physiological_tetras1"] == "grades 1, 1.5"
    assert "function fmt(x, d) { return x.toFixed(d === undefined ? 2 : d); }" in S.PAGE_TEMPLATE
    assert '<span id="time" class="time">0.00 s</span>' in S.PAGE_TEMPLATE


# ---------------------------------------------------------------- the list
def test_list_order_groups_and_chips(S, cat_specs):
    TX = S.TX
    assert [TX.GROUP_NAMES[g] for g in TX.GROUP_ORDER if g != "other"] == TASK_GROUPS
    rows, notes = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    assert notes == {}
    order = []
    for _, text, g, _ in rows:
        if g not in order:
            order.append(g)
    assert [TX.GROUP_NAMES[g] for g in order] == TASK_GROUPS
    series, groups = {}, {}
    for m, text, g, sub in rows:
        series.setdefault(text["series_key"], []).append(text["chip"])
        groups.setdefault(text["series_key"], set()).add((g, sub))
    assert all(len(v) == 1 for v in groups.values()), {k: v for k, v in groups.items() if len(v) > 1}
    for k, chips in series.items():
        assert len(set(chips)) == len(chips), (k, chips)
    exam_keys = {text["series_key"] for m, text, _, _ in rows if m["since"] != "catalogue_v1"}
    cat_keys = {text["series_key"] for m, text, _, _ in rows if m["since"] == "catalogue_v1"}
    assert not exam_keys & cat_keys
    # the walk clips: a gait series lies in 'Gait', the asterixis level 0 beside its grades, the clonus with SCATS
    g_of = {m["clip_id"]: g for m, _, g, _ in rows}
    assert g_of["gait_E8_parkinsonian_g2"] == "gait" and g_of["jerk_asterixis_E2_level0"] == "negative_myoclonus"
    assert g_of["jerk_clonus_J2_scats0"] == "clonus" and g_of["tremor_E3_reach_left_g0"] == "reach"
    # task SKELV3: a specific level 0 beside its grades, the neutral ones in the references, the exam's dystonia
    # with the catalogue's
    assert g_of["invol_E7_flap_level0"] == "stereotypy" and g_of["invol_E4a_mirror_level0"] == "mirror"
    assert g_of["invol_E1_level0"] == "reference" and g_of["post_E7_cd_geste_C1"] == "dystonia"
    assert g_of["dystonia_E7_neck_rotation_left_g2"] == "dystonia" and g_of["post_W_writers_cramp_g1"] == "dystonia"
    assert g_of["post_E7_pkd_choreic"] == "paroxysmal" and g_of["atax_ROMB_sensory_bedside"] == "ataxia"
    assert g_of["park_E7_arise_g2"] == "parkinsonism" and g_of["post_E7_lumbar_open"] == "parkinsonian_posture"
    assert g_of["invol_hemiballism_E1_left_g2"] == "chorea" and g_of["invol_td_E7_thrusting_s3"] == "dyskinesia"
    assert g_of["invol_moving_toes_E1_left_a2"] == "stereotypy" and g_of["freezing_E8_g2_trembling"] == "freezing"
    chip = {m["clip_id"]: text["chip"] for m, text, _, _ in rows}
    summ = {m["clip_id"]: text["summary"] for m, text, _, _ in rows if "summary" in text}
    assert chip["invol_chorea_E1_g2"] == "2 (mild, common)"
    assert chip["invol_chorea_E1_g2b"] == "2 (moderate, intermittent)" and summ["invol_chorea_E1_g2b"] == \
        "grades 1, 2, 3, 4"
    assert chip["park_E7_arise_g1_slow"] == "1 (slow)" and chip["park_E4L_sequence_ps_level2"] == "level 2"
    assert summ["park_E4L_sequence_ps_level2"] == "levels 0, 1, 2, 3"
    assert chip["invol_athetosis_E1_a2_d3"] == "amplitude 2, duration 3"
    assert summ["invol_athetosis_E1_a2_d3"] == "amplitude 1 to 3, duration 2 to 4"
    assert chip["post_E7_cd_rotation_g2_duration4"] == "duration 4" and chip["atax_ROMB_cerebellar"] == "ratio 1.5"
    # task SKELV3 review: one label for a ladder of one kind, the grade shown under another heading named, the
    # BFMDRS grade read as its severity factor
    assert summ["post_E7_cd_rotation_g2_duration4"] == ('duration 1, 2, 4, 5; see also "rotation to the left, '
                                                        'intermittent (duration score 3)"')
    assert chip["post_E7_torsion_dystonia_sev2"] == "severity 2"
    assert summ["post_E7_torsion_dystonia_sev2"] == "severity 2, 3"          # every spec, sev3 included


# ---------------------------------------------------------------- refusal
def test_module_validity_refuses(S):
    t = {"validity": {"limits_new": 0, "gait": {"ok": False}, "finite": True}}
    assert S.module_failures(t) == ["gait"]
    assert S.refusal(t, module_checks=False) == [] and S.refusal(t, module_checks=True) == ["gait"]
    t2 = {"validity": {"limits_new": 3, "gait": {"ok": False}, "finite": False}}
    assert S.refusal(t2, module_checks=True) == ["limits_new", "gait", "finite"]
    assert S.refusal({"validity": None}, True) == [] and S.module_failures(None) == []


@needs_body
def test_refused_clips_listed_not_shown(S, tmp_path):
    from mdfm.synth import exam as EX
    specs = {s.clip_id: s for s in EX.exam_set()}
    base = specs["tremor_E1_rest_g2"]
    body, truth = base.build()
    bad = dict(truth, validity=dict(truth["validity"] or {}, gait={"ok": False}))
    fake = dataclasses.replace(base, clip_id="test_refused_gait", build=lambda: (body, bad))
    out = str(tmp_path / "skeleton_catalogue").replace("\\", "/")
    meta, why = S.export_clip(fake, out, S.SET_CATALOGUE, module_checks=True)
    assert meta is None and why == ["gait"] and not os.path.exists(S.data_path(out, "test_refused_gait"))
    meta, why = S.export_clip(fake, out, S.SET, module_checks=False)          # the exam rule keeps it
    assert why is None and os.path.exists(S.data_path(out, "test_refused_gait"))
    os.remove(S.data_path(out, "test_refused_gait"))
    sel = [specs["tremor_E1_rest_g0"], base]
    for s in sel:
        assert S.export_clip(s, out, S.SET_CATALOGUE, module_checks=True)[1] is None
    refused = {"test_refused_gait": ["gait"]}
    S.write_refused(out, refused)
    assert S.read_refused(out) == refused
    metas, blobs = S.load_all(out, sel + [fake])
    assert [m["clip_id"] for m in metas] == ["tremor_E1_rest_g0", "tremor_E1_rest_g2"]
    label = S.refused_label(fake)
    assert label == f"{base.operator}, {base.grade_text} (clip id test_refused_gait)"
    page, nbytes, total, notes = S.build_page(out, metas, blobs, [], mode="catalogue", refused=refused,
                                              set_name=S.SET_CATALOGUE, refused_labels={"test_refused_gait": label})
    with open(page, encoding="utf-8") as f:
        html = f.read()
    assert notes == {} and html.count('class="clipdata"') == 2 and 'data-clip="test_refused_gait"' not in html
    assert ("<p>Refused clips: 1 clip whose build fails the validity checks (the exam's checks and the module's "
            f"own) is not shown: {label}.") in html
    # the list filter reads the clips' texts, not the series ids; a dystonic-posture level 0 is not tagged
    # 'Normal'; the identifiers are explained; the page's description is the catalogue's
    assert 'b.dataset.text = norm([s.label, s.itemName, s.famName, s.search].join(" "));' in html
    assert "s.key]" not in html
    assert 'id="tagB"' in html and "Dystonic posture only (without the symptom)" in html
    assert "<p>Identifiers:" in html and 'content="SYNTHETIC: joint positions' in html
    assert "built by its specification (symptom,\n        grade, parameters)" in html
    assert "exam specification (operator," not in html
    assert "<title>SYNTHETIC Symptom catalogue, skeleton</title>" in html and "c.en." not in html
    assert "SYNTHETIC" in html and "\u2014" not in html and "\u2013" not in html and "__" not in html.split(
        '<script type="application/json"')[0]
    idx = _index(html)
    assert idx["set"] == "skeleton_v2" and idx["family_order"] == ["tremor"]
    assert all("en" not in c and "fr" not in c for c in idx["clips"])
    assert {c["text"]["family"] for c in idx["clips"]} == {"Tremors"}
    assert not french_left(_page_text(html))
    _, man = S.write_manifest(out, metas, blobs, page, nbytes, [], dict(refused), notes, set_name=S.SET_CATALOGUE,
                              specs=sel + [fake], modules=list(S.DEFAULT_MODULES))
    assert man["refused"] == refused and man["clips_refused_by_family"] == {"tremor": 1}
    assert man["set"] == "skeleton_v2" and man["clips_planned"] == 3 and man["clips_by_group"] == {"tremor": 2}
    assert man["unlabelled"] == {} and "data_from" not in man


# ---------------------------------------------------------------- page_texts never raises, nothing refuses for it
def test_texts_never_raise(S):
    exam_like = {"clip_id": "tremor_E1_unknown_g2", "family": "tremor", "segment": "E1", "segment_title": "",
                 "item": "3.17", "operator": "a new operator", "grade": 2, "grade_text": "MDS-UPDRS 3.17 grade 2",
                 "params_text": "5 Hz", "level0": None, "since": "exam_v2"}
    text, notes = S.page_texts(exam_like)
    assert text["operator"] == "a new operator" and text["series"] == "a new operator"
    assert notes == ["series: tremor_E1_unknown"] and text["segment"] == "E1"
    cat_like = dict(exam_like, clip_id="new_family_E1_x", family="new_family", since="catalogue_v1",
                    grade_text="a new grade text", item="a new item")
    text, notes = S.page_texts(cat_like)
    assert text["grade"] == "a new grade text" and text["item"] == "a new item" and notes == ["series: new_family_E1_x"]
    assert S.TX.group_of(cat_like) == "other" and text["family"] == "Other"
    rows, notes = S.catalogue_rows([cat_like])
    assert rows[0][2] == "other" and rows[0][3] == "other"
    assert notes == {"new_family_E1_x": ["series: new_family_E1_x", "heading: new_family_E1_x"]}


def _fake_meta(clip_id, **kw):
    m = {"clip_id": clip_id, "family": "tremor", "segment": "E1", "segment_title": "", "item": "3.17",
         "operator": "a new operator", "grade": 2, "grade_text": "a new grade text", "params_text": "5 Hz",
         "level0": None, "since": "catalogue_v1", "views": ["front"], "heading_deg": 0.0,
         "floor": {"floor_y_m": 0.0}, "scene_boxes_m": [], "walk": None, "toe_stance": False,
         "_skeleton": {"n_frames": 1, "n_points": 1, "n_joints": 1, "fps": 30, "duration_s": 0.0, "hands": None,
                       "hand_pose": "rest"}}
    m.update(kw)
    return m


def test_catalogue_page_lists_what_fell_back(S, tmp_path):
    """A series without a label is shown under its operator and listed; the page is built (the French-text
    refusal and its --allow-english flag are gone with the French page)."""
    out = str(tmp_path).replace("\\", "/")
    m = _fake_meta("tremor_E1_new_g2")
    page, _, _, notes = S.build_page(out, [m], {m["clip_id"]: b"x"}, [], mode="catalogue", set_name=S.SET_CATALOGUE)
    assert os.path.exists(page) and notes == {"tremor_E1_new_g2": ["series: tremor_E1_new", "heading: tremor_E1_new"]}
    with open(page, encoding="utf-8") as f:
        idx = _index(f.read())
    assert idx["clips"][0]["text"]["series"] == "a new operator"
    with pytest.raises(SystemExit):
        S.main(["--set", "catalogue", "--page-only", "--allow-english", "--list"])


def test_code_deps_follow_the_imports(S):
    """The dependencies of a module are read from the sources (no import): one family's edit leaves the clips of
    the families that do not import it valid, and the families left out of a --modules subset are not followed.
    Since the shared wave, exam.scene_mesh reads postures, ataxia and reach; since the gap wave the families read
    each other, so an edit of any family file stales every clip."""
    synth = os.path.join(paths.TREE, "mdfm", "synth")
    sub = ("tremors", "jerks", "gait")                                   # the SKELV2 subset
    skip = [os.path.join(synth, f"{m}.py") for m in S.blocked_modules(sub)]
    deps = {m: S.code_deps(os.path.join(synth, f"{m}.py"), skip) for m in sub}
    for m in sub:
        assert f"mdfm/synth/{m}.py" in deps[m] and "mdfm/synth/body.py" in deps[m], m
        assert "mdfm/__init__.py" in deps[m] and "mdfm/synth/__init__.py" in deps[m], m
        assert not {f"mdfm/synth/{x}.py" for x in S.blocked_modules(sub)} & set(deps[m]), m
        # since the shared wave (task SKELV4a) every family reads reach.py (exam.scene_mesh draws the examiner's
        # finger of reach.scene_mesh); since the gap wave (task SKELV4b) tremors.py reads gait.py (the rest tremor
        # while walking) and jerks.py (the orthostatic myoclonus's balance), so the three read each other
        assert {f"mdfm/synth/{x}.py" for x in sub} | {"mdfm/synth/reach.py"} <= set(deps[m]), m
    assert "mdfm/body/rotations.py" in deps["gait"] and "mdfm/synth/legcontact.py" in deps["gait"]
    st = S.CodeState([], S.blocked_modules(sub))
    assert set(S.EXPORT_CODE) <= set(st._module(os.path.join(synth, "gait.py"))[1])
    # the seven modules (nothing left out): since the gap wave every family reads every other (exam.scene_mesh routes
    # to postures, ataxia and reach; tremors reads gait and jerks; gait reads involuntary and parkinsonism), so an
    # edit of any family file makes every catalogue clip stale
    full = {m: S.code_deps(os.path.join(synth, f"{m}.py")) for m in S.DEFAULT_MODULES}
    assert S.blocked_modules(S.DEFAULT_MODULES) == []
    for m in S.DEFAULT_MODULES:
        assert {f"mdfm/synth/{x}.py" for x in S.DEFAULT_MODULES} <= set(full[m]), m


@needs_body
def test_catalogue_data_carry_their_code(S, tmp_path):
    from mdfm.synth import exam as EX
    s = {x.clip_id: x for x in EX.exam_set()}["tremor_E1_rest_g0"]
    sub = S.CodeState([s], S.blocked_modules(("tremors", "jerks", "gait"))).of(s)
    assert "mdfm/synth/postures.py" not in sub["deps"]           # exam.scene_mesh calls it, left out of the subset
    code = S.CodeState([s], S.blocked_modules(S.DEFAULT_MODULES)).of(s)
    assert code["module"] == "mdfm.synth.exam" and code["module_file"] == "mdfm/synth/exam.py"
    assert code["module_sha256"] == S.file_sha256(os.path.join(paths.TREE, "mdfm", "synth", "exam.py"))
    assert code["deps_sha256"] == S.deps_sha256(code["deps"]) and "mdfm/synth/exam.py" in code["deps"]
    assert "mdfm/synth/postures.py" in code["deps"] and code["export_format"] == S.EXPORT_FORMAT
    out = str(tmp_path / "cat").replace("\\", "/")
    assert S.export_clip(s, out, S.SET_CATALOGUE, module_checks=True, code=code)[1] is None
    got = S.read_code(S.data_path(out, s.clip_id))
    assert got == code and S.stale_reason(got, code) is None
    assert S.stale_reason(None, code) == "no code record"
    assert S.stale_reason(dict(got, module_sha256="0" * 64), code) == "module_sha256 changed"
    assert S.stale_reason(dict(got, deps_sha256="0" * 64), code) == "deps_sha256 changed"
    assert S.stale_reason(dict(got, synth_code_sha256="0" * 64), code) is None   # another family's edit
    old = {k: v for k, v in got.items() if k != "export_format"}                 # a file of the SKELV2 export
    assert S.stale_reason(old, code) == "export_format changed"
    with gzip.open(S.data_path(out, s.clip_id), "rb") as f:
        assert json.loads(f.read().decode("utf-8"))["overlays"] == []           # the catalogue data carry them
    metas, blobs = S.load_all(out, [s])
    page, nbytes, _, notes = S.build_page(out, metas, blobs, [], mode="catalogue", set_name=S.SET_CATALOGUE)
    _, man = S.write_manifest(out, metas, blobs, page, nbytes, [], {}, notes, set_name=S.SET_CATALOGUE, specs=[s],
                              modules=[], stale={})
    assert "code" not in man and man["page_code"]["synth_code_sha256"] and man["stale"] == {}
    assert man["files"][0]["module_sha256"] == code["module_sha256"]
    assert man["files"][0]["deps_sha256"] == code["deps_sha256"]


def test_series_summary_text_then_grades(S):
    TX = S.TX
    ms = [{"clip_id": "x_E1_posture", "grade": 0, "level0": None},
          {"clip_id": "x_E1_g2", "grade": 2, "level0": "x_E1_posture"},
          {"clip_id": "x_E1_g3", "grade": 3, "level0": "x_E1_posture"}]
    chips = TX.series_chips(ms)
    assert chips == {"x_E1_posture": "posture only", "x_E1_g2": "2", "x_E1_g3": "3"}
    assert TX.series_summary(ms, chips) == "posture only, grades 2, 3"


# ---------------------------------------------------------------- the families left out are never imported
def test_left_out_families_are_blocked(S):
    import importlib
    import sys
    assert S.blocked_modules(S.DEFAULT_MODULES) == []                           # task SKELV3: none by default
    assert S.blocked_modules(("tremors", "jerks", "gait")) == ["involuntary", "postures", "parkinsonism", "ataxia"]
    guard = S.BlockedModules(["no_such_family_for_this_test"])
    assert guard.names == {"mdfm.synth.no_such_family_for_this_test": "no_such_family_for_this_test"}
    sys.meta_path.insert(0, guard)
    try:
        with pytest.raises(S.BlockedModule) as e:
            importlib.import_module("mdfm.synth.no_such_family_for_this_test")
        assert e.value.name == "mdfm.synth.no_such_family_for_this_test"
        assert not isinstance(e.value, ModuleNotFoundError)     # a 'from . import x' does not swallow it
    finally:
        sys.meta_path.remove(guard)
    note = S.deferred_note({"gait_E8_choreic": ["mdfm.synth.involuntary"]},
                           {"gait_E8_choreic": "choreic (dancing) gait (clip id gait_E8_choreic)"})
    assert note == (" 1 clip is not built: its build calls a family of the catalogue still being written "
                    "(involuntary movements), which this catalogue does not load: choreic (dancing) gait (clip id "
                    "gait_E8_choreic).")
    assert S.deferred_note({}) == ""


def test_refusal_notes(S):
    t = S.catalogue_template({"a_g1": ["gait"], "b_g2": [S.BUILD_ERROR + ": ValueError: x"]},
                             {"a_g1": "clip a", "b_g2": "clip b"}, {"c_g0": ["mdfm.synth.ataxia"]})
    assert ("<p>Refused clips: 1 clip whose build fails the validity checks (the exam's checks and the module's own) "
            "is not shown: clip a. 1 clip whose generator stopped on an error is not shown: clip b. 1 clip is not "
            "built: its build calls a family of the catalogue still being written (ataxia), which this catalogue "
            "does not load: c_g0.</p>") in t
    t = S.catalogue_template({})
    assert "<p>Refused clips: No clip failed the validity checks" in t and "__REFUSED_NOTE__" not in t
    assert "__SEGMENTS__" not in t and "breadcrumb the exam segment's" in t
    # task SKELV3 review: the clips the page leaves out although built get their own sentences
    t = S.catalogue_template({"a_g2": [S.INVISIBLE], "b_g3": [S.INVISIBLE], "c_level0": [S.LONE_LEVEL0]},
                             {"a_g2": "clip a", "b_g3": "clip b", "c_level0": "clip c"}, None, ["E1", "E2", "E3", "W"])
    assert ("No clip failed the validity checks (the exam's checks and the module's own). 2 clips whose skeleton "
            "equals their level 0's, frame by frame (the movement reaches no point of the skeleton), are not shown: "
            "clip a; clip b. 1 level 0 clip without a symptom clip on the page is not shown either: clip c.</p>") in t
    assert "breadcrumb (E1 to E3, W) the exam segment's" in t


def test_segment_list(S):
    assert S.segment_list(["E8", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E2f", "E4L", "E4a", "E4aL", "E4b",
                           "TITUB", "J2", "W", "ROMB", "SARA2", "E2", "E7p"]) == \
        "E1 to E8, E2f, E4a, E4aL, E4b, E4L, E7p, J2, ROMB, SARA2, TITUB, W"
    assert S.segment_list(["E1", "E3", "E2f"]) == "E1, E3, E2f" and S.segment_list([]) == ""
    assert S.segment_list([f"E{n}" for n in (*range(1, 11), 19, 21, 22)] + ["E4b"]) == \
        "E1 to E10, E19, E21, E22, E4b"                                  # task SKELV4b: runs of three or more


# ---------------------------------------------------------------- task SKELV3: per-time texts, scenes, exam page
def test_overlays_exported_and_shown_in_the_truths_words(S, tmp_path):
    truth = {"params": {"frame_text": [{"from_s": 0.0, "to_s": 10.0, "text": "EYES OPEN"},
                                       {"from_s": 10.0, "to_s": 20.0, "text": "EYES CLOSED"}],
                        "stimulus_marks": [{"t_s": 5.0, "dur_s": 0.2, "text": "SOUND"},
                                           {"t_s": 9.0, "dur_s": 0.2, "text": "TAP left hand"},
                                           {"t_s": 14.0, "dur_s": 0.2, "text": "TAP right hand"}]}}
    ov = S.clip_overlays(truth)
    assert [o["kind"] for o in ov] == ["frame_text", "frame_text", "stimulus", "stimulus", "stimulus"]
    assert ov[2] == {"kind": "stimulus", "from_s": 5.0, "to_s": 5.2, "text": "SOUND"}
    assert S.clip_overlays({"params": {}}) == [] and S.clip_overlays(None) == []
    shown = S.TX.overlays({"overlays": ov})
    assert [o["text"] for o in shown] == ["EYES OPEN", "EYES CLOSED", "SOUND", "TAP left hand", "TAP right hand"]
    assert shown[0] == {"kind": "frame_text", "from": 0.0, "to": 10.0, "text": "EYES OPEN"}
    assert S.TX.overlays({"overlays": [dict(ov[0], text="A NEW TEXT")]})[0]["text"] == "A NEW TEXT"
    assert S.TX.overlays({}) == []
    # the page: the overlays reach the index as the truth writes them, and the canvas draws them
    m = _fake_meta("atax_ROMB_level0", family="ataxia", segment="ROMB", segment_title="Romberg, feet together",
                   item="Romberg, posturographic ratio", operator="feet together, eyes open then closed (healthy)",
                   grade=0, grade_text="level 0 (healthy): posturographic Romberg ratio 1.5",
                   params_text="eyes open 0 to 10 s, closed 10 to 20 s; normal sway, x 1.5 closed", overlays=ov[:2])
    out = str(tmp_path).replace("\\", "/")
    page, _, _, notes = S.build_page(out, [m], {m["clip_id"]: b"x"}, [], mode="catalogue", set_name=S.SET_CATALOGUE)
    with open(page, encoding="utf-8") as f:
        html = f.read()
    assert notes == {} and "function drawOverlays(ctx, p, D, isA)" in html and "drawOverlays(ctx, p, D, isA);" in html
    c = _index(html)["clips"][0]
    assert [o["text"] for o in c["text"]["overlays"]] == ["EYES OPEN", "EYES CLOSED"]
    assert c["text"]["params"] == "eyes open 0 to 10 s, closed 10 to 20 s; normal sway, x 1.5 closed"
    assert c["text"]["segment"] == "Romberg, feet together" and "<p>Texts on the view:" in html
    assert "/^2(?: |$)/.test(c.text.chip)" in html                  # a series opens on its grade 2
    assert html.index("  // the skeleton, far to near") < html.index("  // overlay\n  badge(ctx, p);\n  drawOverlays(")
    assert "badgeMark" not in html                                  # the badge over the skeleton, clear of the body


@needs_body
def _reference(monkeypatch):
    """The synthetic patient on the SMPL+H reference, behind its flag: the family modules these tests build (jerks,
    ataxia, physio) still read the SMPL+H mesh until their wave of docs/OPEN_BODY_PORT_DESIGN.md ports them."""
    from mdfm.synth import openbody as OB
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")


def test_overlays_of_built_clips(S, tmp_path, monkeypatch):
    """A Romberg clip and a startle clip carry their per-time texts into their data files, on the clip's clock (on
    the reference until waves 3 and 6 port jerks and ataxia)."""
    _reference(monkeypatch)
    from mdfm.synth import catalogue as CAT
    specs = {s.clip_id: s for s in CAT.module_specs("ataxia") + CAT.module_specs("jerks")}
    out = str(tmp_path / "cat").replace("\\", "/")
    st = S.CodeState([], [])
    sel = [specs["atax_ROMB_level0"], specs["jerk_startle_E7_normal"]]
    for s in sel:
        assert S.export_clip(s, out, S.SET_CATALOGUE, module_checks=True, code=st.of(s))[1] is None
    rom, sta = S.load_all(out, sel)[0]
    assert [o["text"] for o in rom["overlays"]] == ["EYES OPEN", "EYES CLOSED"]
    assert rom["overlays"][1]["from_s"] == 10.0
    assert rom["overlays"][1]["to_s"] <= rom["_skeleton"]["duration_s"] + 1e-6
    assert {o["text"] for o in sta["overlays"]} == {"SOUND"} and all(o["kind"] == "stimulus" for o in sta["overlays"])
    assert all(abs(o["to_s"] - o["from_s"] - 0.2) < 1e-9 for o in sta["overlays"])
    for m in (rom, sta):
        text, notes = S.page_texts(m)
        assert notes == [] and [o["text"] for o in text["overlays"]] == [o["text"] for o in m["overlays"]]


def test_scene_parts_keep_boxes_and_list_the_rest(S, monkeypatch):
    import numpy as np
    from mdfm.synth import viewer as V
    b1, b2 = V.box((0, 0, 0), (1, 0.5, 0.25)), V.box((2, 0, 0), (2.5, 1, 1))
    tet_v = np.array([[5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1]], dtype=float)
    tet_f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    v = np.concatenate([b1[0], tet_v, b2[0]])
    f = np.concatenate([b1[1], tet_f + 8, b2[1] + 12])
    monkeypatch.setattr(S.EX, "scene_mesh", lambda kind, body, floor_y: None if kind is None else (v, f))
    boxes, left = S.scene_parts("chair_table", None, 0.0)
    assert boxes == [[0.0, 0.0, 0.0, 1.0, 0.5, 0.25], [2.0, 0.0, 0.0, 2.5, 1.0, 1.0]]
    assert left == [{"vertices": 4, "faces": 4}] and S.scene_parts(None, None, 0.0) == ([], [])


def test_segment_and_scale_names(S):
    """A clip's segment is its own segment title (exam.segment_title); a bare MDS-UPDRS item takes its title, any
    other scale item is shown as the specification writes it."""
    TX = S.TX
    assert TX.segment_name({"segment": "E4aL", "segment_title": "finger tapping, 15 s run"}) == \
        "finger tapping, 15 s run"
    assert TX.segment_name({"segment": "E2", "segment_title": ""}) == "E2"
    assert TX.scale_name("3.16") == "3.16 Kinetic tremor of the hands" and TX.scale_name("MDS-UPDRS 3.13") == \
        "3.13 Posture"
    assert TX.scale_name("3.10 level 0") == "3.10 Gait (level 0)" and TX.scale_name("TETRAS 4a") == "TETRAS 4a"
    assert TX.scale_name("UMRS action myoclonus (left arm)") == "UMRS action myoclonus (left arm)"


@pytest.mark.skipif(not os.path.exists(f"{EXAM_PAGE}/data"), reason="the SKELV exam data are not on this machine")
def test_exam_page_rebuilt_from_its_data(S, tmp_path):
    """The exam page built by this script from the SKELV data is pinned (English since 2026-11-03)."""
    from mdfm.synth import exam as EX
    metas, blobs = S.load_all(EXAM_PAGE, EX.exam_set())
    out = str(tmp_path).replace("\\", "/")
    page, _, _, notes = S.build_page(out, metas, blobs, [])
    with open(page, "rb") as f:
        html = f.read()
    assert hashlib.sha256(html).hexdigest() == EXAM_PAGE_SHA256 and notes == {}
    html = html.decode("utf-8")
    assert not french_left(_page_text(html))
    assert not french_left(json.dumps(_index(html)["clips"], ensure_ascii=False))


# ---------------------------------------------------------------- task SKELV3 review: landmarks, hidden clips, views
@needs_body
def test_landmark_offsets(S):
    """The drawing landmarks: one per kind and side, on the joint of their rule, mirrored between the sides,
    rounded to 5 mm; the toe tip in front of the toe joint, the heel behind and under the ankle, the lateral
    forefoot outside, the pulps beyond the distal joints, the ears to the sides of the head."""
    from mdfm.synth import skeleton as SK
    off = S.landmark_offsets()
    assert tuple(off) == S.CATALOGUE_LANDMARKS and len(off) == 2 * len(S.LANDMARK_RULES)
    ix = {n: i for i, n in enumerate(SK.JOINT_NAMES)}
    for n, (j, o) in off.items():
        side, kind = n.split("_", 1)
        joint = S.LANDMARK_RULES[kind][0]
        assert j == (ix["head"] if joint == "head" else ix[f"{side}_{joint}"]), n
        assert all(abs(x / S.LANDMARK_ROUND_M - round(x / S.LANDMARK_ROUND_M)) < 1e-6 for x in o), n
        other = off[("right_" if side == "left" else "left_") + kind][1]
        assert o == (-other[0] + 0.0, other[1], other[2]), n
    sx = {"left": 1, "right": -1}
    for side in ("left", "right"):
        assert off[f"{side}_toe_tip"][1][2] > 0.05 and off[f"{side}_heel"][1][2] < -0.04
        assert off[f"{side}_heel"][1][1] < -0.02 and sx[side] * off[f"{side}_forefoot_lateral"][1][0] > 0.05
        assert sx[side] * off[f"{side}_index_tip"][1][0] > 0.015 and sx[side] * off[f"{side}_ear"][1][0] > 0.06


def _blob(pos_mm):
    """The bytes of a minimal data file holding positions (N, P, 3) mm."""
    import numpy as np
    q = np.asarray(pos_mm, dtype=np.int64)
    d = q.copy()
    d[1:] -= q[:-1]
    sk = {"n_frames": q.shape[0], "n_points": q.shape[1], "positions_mm_delta": d.reshape(-1).tolist()}
    return gzip.compress(json.dumps({"skeleton": sk}).encode("utf-8"))


def test_hidden_clips(S):
    """A symptom clip whose points equal its level 0's on every common frame is left out (INVISIBLE), a grade-0 or
    an exam clip is not; a catalogue level 0 whose every symptom clip is refused or left out is left out too."""
    import numpy as np
    rest = np.zeros((5, 3, 3), dtype=np.int64)
    moved = rest.copy()
    moved[2, 1, 1] = 4
    m0 = _fake_meta("x_E1_level0", grade=0)
    ms = [m0, _fake_meta("x_E1_same_g2", level0="x_E1_level0"), _fake_meta("x_E1_moved_g2", level0="x_E1_level0"),
          _fake_meta("x_E1_same_g0", grade=0, level0="x_E1_level0"),
          _fake_meta("x_E1_exam_g2", level0="x_E1_level0", since="exam_v2"),
          _fake_meta("y_E1_level0", grade=0), _fake_meta("y_E1_same_g2", level0="y_E1_level0")]
    blobs = {m["clip_id"]: _blob(moved if "moved" in m["clip_id"] else rest[:4 if "same_g2" in m["clip_id"] else 5])
             for m in ms}
    spec = type("Spec", (), {})
    specs = []
    for m in ms + [_fake_meta("y_E1_refused_g3", level0="y_E1_level0")]:
        s = spec()
        s.clip_id, s.level0 = m["clip_id"], m["level0"]
        specs.append(s)
    assert S.hidden_clips(ms, blobs, specs) == {"x_E1_same_g2": [S.INVISIBLE], "y_E1_same_g2": [S.INVISIBLE],
                                                "y_E1_level0": [S.LONE_LEVEL0]}


def test_catalogue_views_and_index(S):
    assert S.catalogue_views("atax_TITUB_g2", ["left", "head"]) == ["head_left", "left", "head"]
    assert S.catalogue_views("invol_mirror_E4a_g2", ["hand_left", "hands"]) == ["hand_left", "hands"]
    assert S.catalogue_views("park_E4aL_sequence_ft_level2", ["front", "hand_left", "hand_left_medial"]) == \
        ["hand_left", "front", "hand_left_medial"]
    assert S.catalogue_views("post_E7_multifocal", ["front", "left", "back"]) == ["front", "left", "back"]
    assert S.catalogue_views("tremor_E1_head_yaw_g2", ["front", "head"]) == ["front", "head", "head_left"]
    for pat, v in S.DEFAULT_VIEW:
        assert v in S.V.VIEWS or v in S.CATALOGUE_VIEWS, (pat, v)
    m = _fake_meta("atax_TITUB_g2", family="ataxia", segment="TITUB", segment_title="seated rest, titubation",
                   item="titubation", operator="truncal titubation, seated", grade=2,
                   grade_text="titubation grade 2 (FTM bands, head)", params_text="3 Hz, head 1.05 cm peak to peak, "
                   "irregular", level0=None, views=["left", "head"])
    idx, notes = S.page_index_catalogue([m], {m["clip_id"]: 1})
    assert notes == {} and idx["clips"][0]["views"] == ["head_left", "left", "head"]
    assert idx["views"]["head_left"]["title"] == "Close-up, head and neck, left side" and idx["views"]["head_left"][
        "region"] == "head" and set(idx["views"]) == set(S.V.VIEWS) | set(S.CATALOGUE_VIEWS)
    assert idx["landmark_bones"] == S.LANDMARK_BONES and idx["region_landmarks"]["body"] == list(
        S.CATALOGUE_LANDMARKS)
    names = set(S.CATALOGUE_LANDMARKS)
    from mdfm.synth import skeleton as SK
    for a, b, st in S.LANDMARK_BONES:
        assert a in names and (b in names or b in SK.JOINT_NAMES) and st in ("foot", "toe", "finger")
    t = S.catalogue_template({})
    for js in ("function fitBands(pts, d, aspect, v, p)", "S.fit = fitBands(pts, d, aspect, v, p);",
               "drawEars(ctx, c, cc, D, CAP);", "drawExtraBone(ctx, c, cc, it, D, CAP);", "var CAP = closeCap();",
               "var rest = handRest(D);", "xb: extraBones(sk.names)", "return withLandmarks(region, D, regionJoints0",
               "labelBox(ctx, VIEWS[S.preset].title", "p.w < NARROW_W ? 54 : 36"):
        assert t.count(js) == 1, js


def test_headings_and_refused_label_wording(S):
    """No year typed in a heading (the year is in the clip's grade text); the Romberg and action dystonia headings;
    the grade named once in a refused clip's label."""
    heads = {k: lab for k, lab, _ in S.TX.SUBGROUPS}
    assert heads["romberg"].startswith("Romberg test") and heads["task_specific"].startswith("Action dystonia")
    assert not any(re.search(r"\d{4}", lab) for lab in heads.values())
    spec = type("Spec", (), {})()
    for k, v in {"clip_id": "post_W_mirror", "family": "dystonia_distribution", "segment": "W",
                 "item": "mirror dystonia", "operator": "mirror dystonia: the right hand while the left writes",
                 "grade": 2, "grade_text": "mirror dystonia", "level0": "post_W_writing_both_level0",
                 "since": "catalogue_v1"}.items():
        setattr(spec, k, v)
    assert S.refused_label(spec) == "mirror dystonia: the right hand while the left writes (clip id post_W_mirror)"


@needs_body
def test_catalogue_landmarks_make_the_toes_visible(S, tmp_path):
    """The moving toes turn the toe block about the foot joint: the 54 points of the skeleton equal the level 0's,
    the big toe tip moves, so that the page shows the clip (hidden_clips leaves it in)."""
    import numpy as np
    from mdfm.synth import catalogue as CAT
    specs = {s.clip_id: s for s in CAT.module_specs("involuntary")}
    out = str(tmp_path / "cat").replace("\\", "/")
    st = S.CodeState([], [])
    sel = [specs["invol_E1_level0"], specs["invol_moving_toes_E1_left_a2"]]
    for s in sel:
        assert S.export_clip(s, out, S.SET_CATALOGUE, module_checks=True, code=st.of(s))[1] is None
    metas, blobs = S.load_all(out, sel)
    a, b = (S.blob_positions(blobs[s.clip_id]) for s in reversed(sel))
    n = min(len(a), len(b))
    names = S.read_data(S.data_path(out, sel[1].clip_id))["skeleton"]["names"]
    assert names[54:] == list(S.CATALOGUE_LANDMARKS) and a.shape[1] == 54 + len(S.CATALOGUE_LANDMARKS)
    toe = names.index("left_toe_tip")
    assert np.array_equal(a[:n, :54], b[:n, :54]) and not np.array_equal(a[:n, toe], b[:n, toe])
    assert S.hidden_clips(metas, blobs, sel) == {}
    lm = S.read_data(S.data_path(out, sel[1].clip_id))["skeleton"]["landmarks"]
    assert lm["left_toe_tip"]["joint"] == names.index("left_foot") and "rule" in lm["left_toe_tip"]
    assert set(lm) == {"head_centre", "nose"} | set(S.CATALOGUE_LANDMARKS)


# ---------------------------------------------------------------- task SKELV4a: the shared wave of the clinical review
SKELV_VIEWS = ("front", "back", "left", "right", "top", "hands", "hands_top", "foot_left", "head", "track_left",
               "examiner", "hand_left", "hand_left_medial")
SKELV_REGIONS = ("body", "hands", "foot_left", "head", "hand_left")


def _titub_meta():
    return _fake_meta("atax_TITUB_g2", family="ataxia", segment="TITUB", segment_title="seated rest, titubation",
                      item="titubation", operator="truncal titubation, seated", grade=2,
                      grade_text="titubation grade 2 (FTM bands, head)",
                      params_text="3 Hz, head 1.05 cm peak to peak, irregular", views=["left", "head_left"])


def test_views_of_the_shared_wave(S, monkeypatch):
    """The exam page keeps the views and regions of task SKELV; the catalogue page shows every view of viewer.VIEWS
    under the viewer's own title (a new view too, no refusal), does not repeat a viewer view, and frames at least two
    points in every view (the toes with their landmarks)."""
    views, regions = S._views_regions()
    assert tuple(views) == SKELV_VIEWS and tuple(regions) == SKELV_REGIONS and S.EXAM_VIEWS == SKELV_VIEWS
    assert {"hand_left_side", "front_left", "head_left", "foot_left_medial", "foot_back", "toes_left"} <= set(S.V.VIEWS)
    assert not set(S.CATALOGUE_VIEWS) & set(S.V.VIEWS)                  # one head_left: the viewer's
    m = _titub_meta()
    idx, notes = S.page_index_catalogue([m], {m["clip_id"]: 1})
    assert notes == {} and set(idx["views"]) == set(S.V.VIEWS) | set(S.CATALOGUE_VIEWS)
    assert idx["clips"][0]["views"] == ["head_left", "left"]           # the titubation opens on the head's side
    for k, v in idx["views"].items():
        assert v["title"] and v["title"][0].isupper() and not french_left(v["title"]) and "\u2014" not in v["title"], k
        reg, lm = idx["regions"][v["region"]], idx["region_landmarks"].get(v["region"], [])
        assert (52 if reg is None else len(reg)) + len(lm) >= 2 and set(lm) <= set(S.CATALOGUE_LANDMARKS), k
    assert idx["views"]["toes_left"]["title"] == "Close-up, left toes, front and left"
    assert idx["views"]["foot_back"]["title"] == "Close-up, both feet, from behind" and idx["regions"]["toes_left"] == [
        10]
    assert idx["views"]["legs_top"]["title"] == "Legs, from above"
    assert S.catalogue_views("tremor_E2_postural_g2", ["front", "left", "top", "hands"]) == \
        ["front", "left", "top", "hands", "hand_left_side"]                # the hand's up and down from its side
    assert S.catalogue_views("tremor_E2w_wing_g2", ["front", "top"]) == ["front", "top"]
    t = S.catalogue_template({})
    assert t.count("return sel.length + (REGION_LM[region] || []).length >= 2 ? sel : all;") == 1
    monkeypatch.setitem(S.V.VIEWS, "new_view", S.V.View("new_view", 10.0, 5.0, title="a new view"))
    idx, notes = S.page_index_catalogue([m], {m["clip_id"]: 1})
    assert notes == {} and idx["views"]["new_view"]["title"] == "A new view"
    assert tuple(S._views_regions()[0]) == SKELV_VIEWS                  # the exam page does not take it


def test_oriented_boxes_drawn_by_their_corners(S, monkeypatch):
    """A box of the scene turned about the vertical (the examiner's finger) is written as its 8 corners in the
    vertex order of viewer.box, an axis-aligned one as [x0, y0, z0, x1, y1, z1]; the page draws the 12 edges of
    viewer.box."""
    import numpy as np
    from mdfm.synth import viewer as V
    v0, f0 = V.box((0, 0, 0), (1, 0.5, 0.25))
    a = np.radians(30.0)
    Rm = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    h, c = np.array([0.01, 0.008, 0.035]), np.array([0.1, 1.2, 0.5])
    vb, fb = V.box(-h, h)
    v1 = c + vb @ Rm.T
    assert S.oriented_box(v1) and S.oriented_box(v0) and not S.oriented_box(v1[:7])
    bent = v1.copy()
    bent[6, 1] += 0.002
    sheared = c + vb @ np.array([[1.0, 0, 0.3], [0, 1, 0], [0, 0, 1]]).T       # a parallelepiped, not a box
    assert not S.oriented_box(bent) and not S.oriented_box(sheared)
    monkeypatch.setattr(S.EX, "scene_mesh", lambda kind, body, floor_y: (np.concatenate([v0, v1]),
                                                                         np.concatenate([f0, fb + 8])))
    boxes, left = S.scene_parts("chair_target", None, 0.0)
    assert left == [] and boxes[0] == [0.0, 0.0, 0.0, 1.0, 0.5, 0.25]
    assert len(boxes[1]) == 8 and np.allclose(boxes[1], v1, atol=1e-4)
    edges = set()
    for tri in fb:
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if int((np.abs(vb[i] - vb[j]) > 1e-12).sum()) == 1:           # a face edge, not its diagonal
                edges.add(tuple(sorted((int(i), int(j)))))
    js = json.loads(re.search(r"var BOX_EDGES = (\[.*?\]\]);", S.CATALOGUE_JS).group(1))
    assert len(edges) == 12 and {tuple(sorted(e)) for e in js} == edges
    assert S.catalogue_template({}).count("if (b.length === 8) { boxEdges(ctx, c, b); return; }") == 1


@needs_body
def test_examiner_finger_is_an_oriented_box(S):
    """The examiner's finger of reach.scene_mesh is a box turned about the vertical (not axis-aligned): the scene
    parts keep it as its corners instead of its larger axis-aligned box."""
    from mdfm.synth import reach as RC
    from mdfm.synth import viewer as V
    for side in ("left", "right"):
        (c, Rm, h), = RC.examiner_boxes(side)
        p = c + V.box(-h, h)[0] @ Rm.T
        lo, hi = p.min(0), p.max(0)
        aabb = {(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])}
        assert S.oriented_box(p) and {tuple(x) for x in p} != aabb, side


@needs_body
def test_supine_clip_on_the_couch(S, tmp_path, monkeypatch):
    """A SARA 8 clip: the room floor and the couch's heading of ataxia.bed_view, the couch as two boxes from the floor
    to COUCH_TOP_M above it, marked supine, opening on the couch views, never following the pelvis (on the reference
    until wave 6 ports ataxia)."""
    _reference(monkeypatch)
    from mdfm.synth import ataxia as AT
    from mdfm.synth import catalogue as CAT
    s = {x.clip_id: x for x in CAT.module_specs("ataxia")}["atax_SARA8_heelshin_g0"]
    code = S.CodeState([s], []).of(s)
    body, truth = s.build()
    fake = dataclasses.replace(s, build=lambda: (body, truth))
    out = str(tmp_path / "cat").replace("\\", "/")
    meta, bad = S.export_clip(fake, out, S.SET_CATALOGUE, module_checks=True, code=code)
    bv = AT.bed_view(body)
    assert bad is None and meta["supine"] is True and meta["heading_deg"] == bv["heading_deg"] == 0.0
    assert abs(meta["floor"]["floor_y_m"] - bv["floor_y"]) < 1e-12 and "bed_view" in meta["floor"]["rule"]
    assert meta["floor"]["lowest_joint_median_m"] > bv["floor_y"] + 0.5           # the joints lie on the couch
    boxes = meta["scene_boxes_m"]
    assert len(boxes) == 2 and all(len(b) == 6 for b in boxes)
    assert abs(min(b[1] for b in boxes) - bv["floor_y"]) < 1e-3
    assert abs(max(b[4] for b in boxes) - bv["floor_y"] - AT.COUCH_TOP_M) < 1e-3
    metas, blobs = S.load_all(out, [s])
    idx, notes = S.page_index_catalogue(metas, {s.clip_id: len(blobs[s.clip_id])})
    c = idx["clips"][0]
    assert notes == {} and c["supine"] is True and c["views"][:2] == ["legs_left", "legs_top"]
    assert "track_left" not in c["views"] and idx["views"]["legs_top"]["region"] == "legs"
    assert S.catalogue_views("atax_SARA8_heelshin_right_g0", ["right", "top", "left"]) == \
        ["legs_right", "legs_top", "right", "top", "left"]
    t = S.catalogue_template({})
    for js in ('c.views.indexOf("track_left") >= 0 && !c.supine', "supine: !!d.supine", "var SY = shadowY(D);",
               "SY + 0.002, P[o + 3 * a + 2]"):
        assert t.count(js) == 1, js


def test_shared_wave_places_and_chips(S, cat_specs):
    """The SARA 8 clips, the touching tics and the scissoring gait under their headings, with their chips and list
    lines and their specifications' own texts; the spastic gait's grades mild before moderate."""
    by = {s.clip_id: s for s in cat_specs}
    rows, notes = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    assert notes == {}
    r = {m["clip_id"]: (text, g, sub) for m, text, g, sub in rows}
    text, g, sub = r["atax_SARA8_heelshin_g2"]
    assert g == "ataxia" and sub == "heel_shin" and text["series_key"] == "atax_SARA8_heelshin"
    assert text["operator"] == by["atax_SARA8_heelshin_g2"].operator
    assert text["title"] == "Heel-shin slide (SARA item 8): left heel on the right shin, supine"
    assert text["segment"] == "heel-shin slide, supine" and text["grade"] == by["atax_SARA8_heelshin_g2"].grade_text
    assert text["scale"] == "SARA 8" and text["chip"] == "2"
    assert text["summary"] == "grades 0, 1, 2, 3, 4"
    t0 = r["atax_SARA8_heelshin_right_g0"][0]
    assert t0["chip"] == "level 0" and t0["grade"] == by["atax_SARA8_heelshin_right_g0"].grade_text
    for cid, chip in (("jerk_tics_E1_touch_c2", "complexity 2"), ("jerk_tics_E1_touch_c3", "complexity 3"),
                      ("jerk_tics_E7_touch_c2", "complexity 2")):
        text, g, sub = r[cid]
        assert g == "tics" and sub == "complex_tics" and text["chip"] == chip, cid
        assert text["params"] == by[cid].params_text, cid
    assert r["jerk_tics_E1_touch_c3"][0]["summary"] == "complexity 2, 3"
    order = [m["clip_id"] for m, text, _, _ in rows if text["series_key"] == "gait_E8_spastic_diplegic"]
    assert order == ["gait_E8_spastic_diplegic_mild", "gait_E8_spastic_diplegic"]
    mild, mod = r[order[0]][0], r[order[1]][0]
    assert mild["chip"] == "mild (described level)" and mod["chip"] == "moderate (described level)"
    assert mild["summary"] == "mild, moderate" and mild["series"] == "spastic diplegic gait with scissoring"
    assert mild["operator"] == by[order[0]].operator and mod["params"] == by[order[1]].params_text


def test_exam_v5_tremor_texts(S):
    """The exam tremor texts after the shared wave pass as they are; the series of the postural tremor names the
    hand for the exam_v5 operator and the forearm for the text of the skeleton_v1 data."""
    rest = {"clip_id": "tremor_E1_rest_g4", "family": "tremor", "segment": "E1", "segment_title": "seated rest",
            "item": "3.17", "operator": "tremor, rest, left forearm", "grade": 4,
            "grade_text": "MDS-UPDRS 3.17 grade 4",
            "params_text": "4.5 Hz (module), 12.00 cm peak to peak at the wrist (D4; module level), middle fingertip "
                           "21.03 cm, centred 5.5 deg toward flexion (clear of the chair)",
            "level0": "tremor_E1_rest_g0", "since": "exam_v5"}
    text, notes = S.page_texts(rest)
    assert notes == [] and text["params"] == rest["params_text"] and text["series"] == "rest, left forearm"
    assert text["family"] == "Tremor" and text["item"] == "3.17 Rest tremor amplitude" and text["chip"] == "4"
    post = dict(rest, clip_id="tremor_E2_postural_g3", segment="E2", segment_title="arms outstretched", item="3.15",
                operator="tremor, postural, left hand", grade=3, grade_text="MDS-UPDRS 3.15 grade 3",
                params_text="5 Hz (module), 5.48 cm peak to peak at the middle fingertip, the hand at the wrist, the "
                            "shoulder joining (angle weight 0.25)", level0="tremor_E2_postural_g0")
    text, notes = S.page_texts(post)
    assert notes == [] and text["operator"] == "tremor, postural, left hand" and text["series"] == \
        "postural, left hand"
    text, _ = S.page_texts(dict(post, operator="tremor, postural, left forearm"))     # a skeleton_v1 text
    assert text["series"] == "postural, left forearm"
    text, _ = S.page_texts(dict(post, clip_id="tremor_E1_rest_g1b", grade=1))
    assert text["chip"] == "1 (below the estimator floor)"


@needs_body
def test_exam_v5_tremor_texts_built(S):
    """The parameter texts of built exam tremor clips (exam.describe on their truth) pass as they are."""
    from mdfm.synth import exam as EX
    specs = {s.clip_id: s for s in EX.exam_set()}
    for cid in ("tremor_E1_rest_g4", "tremor_E2_postural_g3", "tremor_E1_constancy_g1", "tremor_E2_reemergent_g2"):
        s = specs[cid]
        _, truth = s.build()
        described = EX.describe(s, truth)
        text, notes = S.page_texts(spec_meta(s, described))
        assert not notes and text["params"] == described and not french_left(described), (cid, described)


def test_leg_contact_reported_not_refused(S):
    """The gaits' leg-to-leg check (validity 'legs') is written into the data and never refuses on its own: since
    2026-11-03 it enters gait.ok, so a failing gait clip is refused once, through 'gait'."""
    t = {"validity": {"gait": {"ok": True}, "legs": {"ok": False, "hard_min_m": -0.02}, "finite": True}}
    assert S.module_failures(t) == [] and S.refusal(t, module_checks=True) == []
    assert S.reported_failures(t) == ["legs"] and S.reported_failures({"validity": {}}) == []
    t["validity"]["gait"]["ok"] = False
    assert S.refusal(t, module_checks=True) == ["gait"]


# ---------------------------------------------------------------- task SKELV4b: the gap wave of the symptom catalogue
GAP_PLACES = {   # a clip of every new family or series: (group, heading key)
    "park_E21_applause_g2": ("atypical_parkinsonism", "applause"),
    "park_E7_arise_abrupt": ("atypical_parkinsonism", "arise_abrupt"),
    "park_E22_sitdown_enbloc": ("atypical_parkinsonism", "sitdown_enbloc"),
    "park_E22_sitdown_level0": ("atypical_parkinsonism", "sitdown_enbloc"),
    "park_E1_motorstate_offon": ("parkinsonism", "motor_fluctuations"),
    "park_E7_gowers": ("gait", "gowers"),
    "gait_E8_psp": ("gait", "psp_gait"), "gait_E8_dromedary": ("gait", "dystonic_gait"),
    "gait_E8_vestibular_left_eyes_open": ("gait", "vestibular_gait"),
    "gait_E19_vestibular_left_eyes_closed": ("gait", "vestibular_gait"),
    "gait_E19_sensory_ataxic_eyes_closed": ("gait", "sensory_ataxic_gait"),
    "gait_E8_functional_dragging_left": ("gait", "functional_gait"),
    "gait_E8_fog_shuffling_g3": ("freezing", "fog_shuffling"),
    "jerk_functional_E1_axial_sev2_dur2": ("myoclonus", "functional_jerks"),
    "jerk_functional_E7_axial_sev2_dur2": ("myoclonus", "functional_jerks"),
    "orthostatic_myoclonus_E7_present": ("myoclonus", "orthostatic_myoclonus"),
    "jerk_tonic_tics_E1_f2_i3_mixed": ("tics", "tonic_tics"),
    "invol_lid_dystonic_E1_s3": ("dyskinesia", "lid_dystonic"),
    "invol_hd_dystonia_E7_g3": ("chorea", "hd_dystonia"),
    "invol_td_stereotypy_E7_marching_s3": ("stereotypy", "tardive_stereotypy"),
    "invol_td_stereotypy_E1_rubbing_s3": ("stereotypy", "tardive_stereotypy"),
    "tremor_E3_essential_terminal_tetras2": ("tremor", "essential"),
    "tremor_E1_essential_rest_g2": ("tremor", "essential"),
    "tremor_E4_kinetic_ps_tetras2": ("tremor", "kinetic_tetras"), "tremor_E1_head_mixed_g2": ("tremor", "head"),
    "isometric_tremor_E4b_fist_present": ("tremor", "isometric"), "tremor_E1_leg_rest_g2": ("tremor", "leg_rest"),
    "tremor_E8_walk_rest_g2": ("tremor", "walk_rest"), "standing_tremor_E10_tetras9_g2": ("tremor", "standing_tremor"),
}


def test_gap_wave_places_and_chips(S, cat_specs):
    """The gap wave's clips under their clinical groups and headings, with their chips and list lines; the atypical
    parkinsonian syndromes right after the other parkinsonian signs; the dual task in the truth's words."""
    TX = S.TX
    assert TX.GROUP_ORDER.index("atypical_parkinsonism") == TX.GROUP_ORDER.index("parkinsonism") + 1
    assert TX.GROUP_NAMES["atypical_parkinsonism"] == "Atypical parkinsonism (PSP)"
    by = {s.clip_id: s for s in cat_specs}
    rows, notes = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    assert notes == {}
    r = {m["clip_id"]: (text, g, sub) for m, text, g, sub in rows}
    for cid, (g, sub) in GAP_PLACES.items():
        assert r[cid][1:] == (g, sub), (cid, r[cid][1:])
        # the specification's own texts, without their marks of provenance (wave SH1d: the provenance line)
        assert r[cid][0]["operator"] == TX.split_provenance(by[cid].operator)[0], cid
        assert r[cid][0]["grade"] == TX.split_provenance(by[cid].grade_text)[0], cid
    heads = {k: lab for k, lab, _ in TX.SUBGROUPS}
    assert heads["applause"] == "Applause sign" and heads["sitdown_enbloc"] == "Sitting down en bloc"
    assert heads["arise_abrupt"] == "Abrupt arising" and heads["gowers"].startswith("Arising with Gowers' manoeuvre")
    assert heads["psp_gait"] == "Gait of progressive supranuclear palsy (PSP)"
    assert heads["vestibular_gait"] == "Vestibular gait with veering"
    assert heads["sensory_ataxic_gait"] == "Sensory ataxic gait (stamping)"
    assert heads["tonic_tics"] == "Tonic tics (dystonic)" and heads["tardive_stereotypy"].startswith(
        "Tardive stereotypies")
    assert heads["functional_jerks"].startswith("Functional axial jerks")
    chip = {m["clip_id"]: text["chip"] for m, text, _, _ in rows}
    summ = {m["clip_id"]: text["summary"] for m, text, _, _ in rows if "summary" in text}
    assert [chip[f"park_E21_applause_g{g}"] for g in range(4)] == ["score 0", "score 1", "score 2", "score 3"]
    assert summ["park_E21_applause_g1"] == "score 0, 1, 2, 3"
    assert chip["park_E22_sitdown_level0"] == "level 0" and chip["park_E22_sitdown_enbloc"] == "en bloc"
    assert chip["park_E1_motorstate_offon"] == "OFF to ON" and chip["park_E1_motorstate_level0"] == "level 0"
    assert chip["gait_E8_vestibular_left_eyes_open"] == "eyes open"
    assert chip["gait_E19_vestibular_left_eyes_closed"] == "eyes closed"
    assert chip["jerk_functional_E1_axial_sev3_dur2"] == "severity 3, duration 2"
    assert summ["jerk_functional_E1_axial_sev3_dur2"] == "severity 2 to 3, duration 1 to 2"
    fun = [m["clip_id"] for m, text, _, _ in rows if text["series_key"] == "gait_E8_functional"]
    assert len(fun) == 6 and r[fun[0]][0]["series"] == "clinical presentations"
    assert len({chip[c] for c in fun}) == 6
    ov = TX.overlays({"overlays": [{"kind": "frame_text", "from_s": 11.0, "to_s": 21.0,
                                    "text": "DUAL TASK: COUNTING BACKWARD BY SEVENS ALOUD"}]})
    assert ov[0]["text"] == "DUAL TASK: COUNTING BACKWARD BY SEVENS ALOUD"


def test_gap_wave_page_settings(S):
    """The page embeds the 540 clips (a local file, never published); the per-time texts shrink to fit the view;
    the about section names the new group and scales."""
    assert S.EMBED_LIMIT_BYTES == 25_000_000
    assert "var fs = Math.max(9, Math.min(14, 14 * (p.w - 36) / ctx.measureText(o.text).width));" in S.OVERLAY_JS
    t = S.catalogue_template({})
    assert "atypical parkinsonism (PSP)" in t and "PSPRS, S-FMDRS" in t and "(DUAL TASK)" in t


# ---------------------------------------------------------------- --data-from: a preview beside the owner's folders
@needs_body
def test_page_only_from_another_folder(S, tmp_path, monkeypatch):
    """--page-only --data-from reads the data and refusals of a folder and writes the page, the manifest and the
    refusals to --out only; no run record; refused without --page-only or when the two folders are one."""
    from mdfm.synth import exam as EX
    root = str(tmp_path / "viewer").replace("\\", "/")
    monkeypatch.setattr(S.V, "VIEWER_ROOT", root)
    src, out = f"{root}/data_set", f"{root}/preview"
    sel = [s for s in EX.exam_set() if s.clip_id in ("tremor_E1_rest_g0", "tremor_E1_rest_g2")]
    for s in sel:
        assert S.export_clip(s, src)[1] is None
    before = sorted(os.listdir(src)), sorted(os.listdir(f"{src}/{S.DATA}"))
    monkeypatch.setattr(S, "set_specs", lambda set_name, modules: sel)
    monkeypatch.setattr(S, "write_record", lambda *a, **k: pytest.fail("run record written by a preview"))
    S.main(["--page-only", "--out", out, "--data-from", src])            # without --no-record: still no record
    assert (sorted(os.listdir(src)), sorted(os.listdir(f"{src}/{S.DATA}"))) == before
    assert os.path.exists(f"{out}/{S.PAGE}") and os.path.exists(f"{out}/{S.MANIFEST}")
    with open(f"{out}/{S.MANIFEST}", encoding="utf-8") as f:
        man = json.load(f)
    assert man["data_from"] == src and man["clips"] == 2
    with pytest.raises(SystemExit, match="--page-only only"):
        S.main(["--out", out, "--data-from", src])
    with pytest.raises(SystemExit, match="output folder itself"):
        S.main(["--page-only", "--out", out, "--data-from", out])


# ---------------------------------------------------------------- wave SH1d: the page after the clinical review
def test_floor_over_the_standing_frames(S, monkeypatch):
    """The catalogue export's floor: the lowest body joint's minimum over the frames where the feet stand still,
    the median rule without enough of them; the exam export keeps the median rule."""
    import numpy as np
    monkeypatch.setattr(S.V, "sole_offset_m", lambda: 0.018)
    n = 90
    pos = np.zeros((n, 22, 3))
    pos[:, :, 1] = 0.5
    pos[:, list(S.FOOT_JOINTS), 1] = 0.020                  # feet on the floor, still for the first second
    pos[30:, 10, 2] = np.linspace(0.0, 1.0, n - 30)          # then the left toe joint walks off at 1 m/s
    pos[30:, 10, 1] = 0.020 + 0.010 * np.sin(np.linspace(0, 6 * np.pi, n - 30)) ** 2
    pos[10, 10, 1] = 0.019                                   # one still frame 1 mm lower than the rest
    skel = type("Skel", (), {"positions": pos, "fps": 30.0})()
    st = S.standing_frames(pos, 30.0)
    assert st[:29].all() and not st[32:].any()
    fl = S.clip_floor(skel, "standing")
    assert abs(fl["floor_y_m"] - (0.019 - 0.018)) < 1e-12 and fl["standing_frames"] == int(st.sum())
    assert "still" in fl["rule"] and fl["lowest_joint_standing_min_m"] == 0.019
    med = S.clip_floor(skel)
    assert med["floor_y_m"] == float(np.median(pos[:, :22, 1].min(1))) - 0.018 and "standing_frames" not in med
    moving = pos.copy()
    moving[:, 10, 2] = np.linspace(0.0, 3.0, n)              # never still: the median rule
    fl2 = S.clip_floor(type("Skel", (), {"positions": moving, "fps": 30.0})(), "standing")
    assert fl2["floor_y_m"] == float(np.median(moving[:, :22, 1].min(1))) - 0.018 and "fewer than" in fl2["rule"]
    with pytest.raises(ValueError):
        S.clip_floor(skel, "lowest")


def test_event_spans_of_the_jerks(S):
    truth = {"params": {"events": [
        {"onset_s": 2.0, "parts": [{"delay_s": 0.0, "rise_s": 0.05, "hold_s": 0.0, "fall_s": 0.1},
                                   {"delay_s": 0.03, "rise_s": 0.05, "fall_s": 0.2}]},
        {"onset_s": 5.0, "parts": [{"rise_s": 0.1, "fall_s": 0.1}]},
        {"onset_s": 7.0, "manifestation": "trembling"},       # another family's events: not a jerk
        {"context": "turn"}]}}
    assert S.event_spans(truth) == [[2.0, 2.28], [5.0, 5.2]]
    assert S.event_spans({"params": {}}) == [] and S.event_spans(None) == []
    assert S.event_spans({"params": {"ballistic_intervals_s": [[1.0, 3.5]]}}) == [[1.0, 3.5]]     # hemiballism


def _meta_pair(scene="chair", **kw):
    root = _fake_meta("x_E1_level0", grade=0, scene=scene, floor={"floor_y_m": -0.0182},
                      scene_boxes_m=[[0.0, -0.0182, 0.0, 0.5, 0.45, 0.5]])
    mid = _fake_meta("x_E1_posture", grade=0, level0="x_E1_level0", scene=scene, floor={"floor_y_m": -0.0137},
                     scene_boxes_m=[[0.0, -0.0137, 0.0, 0.5, 0.45, 0.5]])
    top = _fake_meta("x_E1_g2", level0="x_E1_posture", scene=scene, floor={"floor_y_m": 0.0045},
                     scene_boxes_m=[[0.0, 0.0045, 0.0, 0.5, 0.45, 0.5]], **kw)
    return root, mid, top


def test_clip_drawn_on_its_level0s_floor(S):
    """PLAN M5: a clip and its level 0 (through a chain of level 0s) share the floor and the chair the page draws,
    read from the index; a clip whose level 0 is not shown, or whose scene differs, keeps its own."""
    root, mid, top = _meta_pair()
    other = _fake_meta("y_E1_g2", level0="y_E1_level0", floor={"floor_y_m": 0.01}, scene="chair",
                       scene_boxes_m=[[1.0, 0.01, 0.0, 1.5, 0.45, 0.5]])
    odd = _fake_meta("x_E1_odd_g3", level0="x_E1_level0", scene="chair_armless", floor={"floor_y_m": 0.002},
                     scene_boxes_m=[[2.0, 0.002, 0.0, 2.5, 0.45, 0.5]])
    metas = [root, mid, top, other, odd]
    assert S.page_roots(metas) == {"x_E1_level0": "x_E1_level0", "x_E1_posture": "x_E1_level0",
                                   "x_E1_g2": "x_E1_level0", "y_E1_g2": "y_E1_g2", "x_E1_odd_g3": "x_E1_level0"}
    idx, _ = S.page_index_catalogue(metas, {m["clip_id"]: 1 for m in metas})
    c = {x["id"]: x for x in idx["clips"]}
    for cid in ("x_E1_level0", "x_E1_posture", "x_E1_g2"):
        assert c[cid]["floor"] == -0.0182 and c[cid]["boxes"] == root["scene_boxes_m"], cid
    assert c["y_E1_g2"]["floor"] == 0.01 and c["y_E1_g2"]["boxes"] == other["scene_boxes_m"]
    assert c["x_E1_odd_g3"]["floor"] == 0.002 and c["x_E1_odd_g3"]["boxes"] == odd["scene_boxes_m"]
    rec = S.floor_record(metas)
    assert rec["x_E1_g2"] == {"root": "x_E1_level0", "own_floor_mm": 4.5, "floor_mm": -18.2, "boxes_shift_mm": 0.0}
    assert rec["x_E1_odd_g3"]["boxes_shift_mm"] == 0.0                  # another scene: drawn with its own chair
    shifted = dict(top, scene_boxes_m=[[0.052, 0.0045, 0.0, 0.552, 0.45, 0.5]])   # a seat placed 52 mm aside
    assert S.floor_record([root, mid, shifted])["x_E1_g2"]["boxes_shift_mm"] == 52.0
    assert S.boxes_shift_mm([[0, 0, 0, 1, 1, 1]], [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]) is None
    assert set(rec) == {"x_E1_posture", "x_E1_g2", "x_E1_odd_g3"} and rec["x_E1_odd_g3"]["floor_mm"] == 2.0
    t = S.catalogue_template({})
    assert t.count("floor: CLIPS.get(id).floor,") == 1 and t.count("boxes: CLIPS.get(id).boxes,") == 1
    assert "floor: d.floor.floor_y_m" not in t and "boxes: d.scene_boxes_m" not in t
    assert "floor: d.floor.floor_y_m" in S.PAGE_TEMPLATE                  # the exam page keeps the data file's
    assert "is drawn on its level 0's floor and with its level 0's chair" in t


def test_reach_level0_without_a_symptom_clip_hidden(S, cat_specs):
    """A catalogue level 0 that no clip names as its level 0 is hidden (the right-arm reach, every reach clip being
    left-sided), but those kept on purpose (the right side of the heel-shin test)."""
    users = {}
    for s in cat_specs:
        if s.level0:
            users.setdefault(s.level0, []).append(s.clip_id)
    alone = sorted(s.clip_id for s in cat_specs if s.since == "catalogue_v1" and s.level0 is None and s.grade == 0
                   and not users.get(s.clip_id))
    assert alone == ["atax_SARA8_heelshin_right_g0", "tremor_E3_reach_right_g0"]
    assert set(S.TX.STANDALONE_LEVEL0) == {"atax_SARA8_heelshin_right_g0"}
    import numpy as np
    ms = [_fake_meta(c, grade=0) for c in alone]
    ms += [_fake_meta("tremor_E3_reach_left_g0", grade=0), _fake_meta("t_g2", level0="tremor_E3_reach_left_g0")]
    blobs = {m["clip_id"]: _blob(np.ones((2, 1, 3)) if m["clip_id"] == "t_g2" else np.zeros((2, 1, 3))) for m in ms}
    spec = type("Spec", (), {})
    specs = []
    pairs = [(c, None) for c in alone] + [("tremor_E3_reach_left_g0", None), ("t_g2", "tremor_E3_reach_left_g0")]
    for cid, l0 in pairs:
        x = spec()
        x.clip_id, x.level0 = cid, l0
        specs.append(x)
    assert S.hidden_clips(ms, blobs, specs) == {"tremor_E3_reach_right_g0": [S.LONE_LEVEL0]}


def test_tetras_heading_of_the_simple_kinetic_tremor(S, cat_specs):
    """The simple kinetic tremor during pronation-supination is graded on the TETRAS bands: its own heading, not
    3.16 (the finger-to-nose item)."""
    rows, notes = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    assert notes == {}
    r = {m["clip_id"]: (text, g, sub) for m, text, g, sub in rows}
    heads = {k: lab for k, lab, _ in S.TX.SUBGROUPS}
    for cid in ("tremor_E4_kinetic_ps_tetras2", "tremor_E4_kinetic_ps_tetras3"):
        text, g, sub = r[cid]
        assert (g, sub) == ("tremor", "kinetic_tetras") and "TETRAS" in heads[sub] and "3.16" not in heads[sub]
        assert text["title"] == ("Simple kinetic tremor (graded on the TETRAS upper-limb bands): left arm, during "
                                 "pronation-supination")
    assert r["tremor_E3_kinetic_g2"][2] == "kinetic" and heads["kinetic"] == "Kinetic tremor (MDS-UPDRS 3.16)"


# ---------------------------------------------------------------- wave SH1d: the provenance line
PROV_CASES = [   # (a specification's text, its clinical text, [(about, mark, kind)])
    ("4.5 Hz (module), 12.00 cm peak to peak at the wrist (D4; module level), middle fingertip 21.03 cm",
     "4.5 Hz, 12.00 cm peak to peak at the wrist, middle fingertip 21.03 cm",
     [("4.5 Hz", "module", "module"), ("12.00 cm peak to peak at the wrist", "D4", "rule_d4"),
      ("12.00 cm peak to peak at the wrist", "module level", "module")]),
    ("15 deg, perpendicular method, coronal (1.5 x the 10 deg cut-off [module]), held from the first frame; knees 5 "
     "deg [module]; feet planted",
     "15 deg, perpendicular method, coronal (1.5 x the 10 deg cut-off), held from the first frame; knees 5 deg; feet "
     "planted", [("1.5 x the 10 deg cut-off", "module", "module"), ("knees 5 deg", "module", "module")]),
    ("UDysRS part 3 step 2 (moderate), type dystonic, declared map; face not shown",
     "UDysRS part 3 step 2 (moderate), type dystonic; face not shown", [("grade", "declared map", "declared_map")]),
    ("declared ladder grade 2 (fingers, wrist)", "grade 2 (fingers, wrist)",
     [("grade", "declared ladder", "declared_ladder")]),
    ("level 0 (cited phases, trunk solved for balance)", "level 0 (trunk solved for balance)",
     [("level 0 (phases)", "cited phases", "cited")]),
    ("WCRS (module item scores)", "WCRS", [("WCRS (item scores)", "module item scores", "module")]),
    ("a tremor rising over the last 10 cm (module envelope)", "a tremor rising over the last 10 cm",
     [("a tremor rising over the last 10 cm (envelope)", "module envelope", "module")]),
    ("2 slower reaches of 3.46 s (module pace, on the level 0's clock)",
     "2 slower reaches of 3.46 s (on the level 0's clock)",
     [("2 slower reaches of 3.46 s (pace)", "module pace", "module")]),
    ("10 cycles at the cited normal rate, 3.67 Hz, no criterion; level-0 amplitude 68.1 mm at 3.67 Hz (cited normal)",
     "10 cycles at the normal rate, 3.67 Hz, no criterion; level-0 amplitude 68.1 mm at 3.67 Hz",
     [("the normal rate", "cited", "cited"), ("level-0 amplitude 68.1 mm at 3.67 Hz", "cited normal", "cited")]),
    ("2.15 Hz, 0.585 of the cited normal rate", "2.15 Hz, 0.585 of the normal rate",
     [("the normal rate", "cited", "cited")]),
    ("MDS-UPDRS 3.17 grade 1 (below the estimator floor)", "MDS-UPDRS 3.17 grade 1",
     [("grade", "below the estimator floor", "estimator_below")]),
    ("30 and 100 degrees (video draft starting posture)", "30 and 100 degrees",
     [("30 and 100 degrees", "video draft starting posture", "draft")]),
    ("fixed from the first frame, no change: ring and little fingers flexed most, the wrist flexed [module]",
     "fixed from the first frame, no change: ring and little fingers flexed most, the wrist flexed",
     [("ring and little fingers flexed most, the wrist flexed", "module", "module")]),
    ("MDS-UPDRS 3.17 grade 3 to grade 1: OFF to ON (the time course compressed, module)",
     "MDS-UPDRS 3.17 grade 3 to grade 1: OFF to ON (the time course compressed)", [("OFF to ON", "module", "module")]),
]


def test_provenance_split(S):
    TX = S.TX
    for text, clean, notes in PROV_CASES:
        got, n = TX.split_provenance(text)
        assert got == clean, (text, got)
        assert [(x["about"], x["mark"], x["kind"]) for x in n] == notes, (text, n)
        assert not re.search(r"\bmodule\b|\bD4\b|declared|estimator|video draft|\bcited\b|\[", got), got
        for about, _, _ in notes:                            # the phrase stays grammatical: carried words bracketed
            assert not re.search(r"\b(envelope|pace|phases|item scores|clock)$", about), about
    for plain in ("5 Hz, 1.73 cm peak to peak at the middle fingertip (band 1 to 3 cm)", "10 cycles at 3.67 Hz",
                  "attacks of 5 to 30 min (Lance 1977, via Latorre and Bhatia)", "a reach timed (on the level 0's "
                  "clock)", ""):
        assert TX.split_provenance(plain) == (plain, [])
    assert set(TX.PROVENANCE_KINDS) == {k for _, k in TX.PROVENANCE_PARTS}
    for k, v in TX.PROVENANCE_KINDS.items():
        assert v and not french_left(v) and not re.search(r"\bmodule\b|_", v), (k, v)


def test_provenance_line_of_the_page(S, tmp_path):
    """The catalogue page shows the clinical texts and lists the provenance under the caption, behind a chip; the
    estimator-floor chips in clinical words; the about section explains the line once."""
    TX = S.TX
    text = {"operator": "tremor, rest, left forearm", "grade": "MDS-UPDRS 3.17 grade 1 (below the estimator floor)",
            "params": "5 Hz (module), 0.58 cm peak to peak at the wrist (D4), middle fingertip 1.01 cm",
            "chip": "1 (below the estimator floor)"}
    out, prov = TX.clinical_texts(text)
    assert out["grade"] == "MDS-UPDRS 3.17 grade 1" and out["chip"] == "1 (smaller amplitude)"
    assert out["params"] == "5 Hz, 0.58 cm peak to peak at the wrist, middle fingertip 1.01 cm"
    assert [p["about"] for p in prov] == ["grade", "5 Hz", "0.58 cm peak to peak at the wrist"]
    assert prov[0]["meaning"] == TX.PROVENANCE_KINDS["estimator_below"] and prov[2]["marks"] == ["D4"]
    assert TX.clinical_chip("1 (above the estimator floor)") == "1 (larger amplitude)"
    m = _fake_meta("tremor_E1_new_g2", item="3.17", operator="tremor, rest, left forearm",
                   grade_text="MDS-UPDRS 3.17 grade 2", params_text="5 Hz (module), 1.73 cm peak to peak at the "
                   "wrist (D4)")
    out_dir = str(tmp_path).replace("\\", "/")
    page, _, _, _ = S.build_page(out_dir, [m], {m["clip_id"]: b"x"}, [], mode="catalogue", set_name=S.SET_CATALOGUE)
    with open(page, encoding="utf-8") as f:
        html = f.read()
    c = _index(html)["clips"][0]
    assert c["text"]["params"] == "5 Hz, 1.73 cm peak to peak at the wrist"
    assert [p["about"] for p in c["text"]["provenance"]] == ["5 Hz", "1.73 cm peak to peak at the wrist"]
    assert html.count("cap.append(provenanceLine(c.text.provenance));") == 1 and "function provenanceLine(pv)" in html
    assert "details.prov{" in html and "<p>Provenance: the captions give the clinical content." in html
    assert "The marks [module]" not in html and "\u2014" not in html and "\u2013" not in html
    assert not french_left(_page_text(html))


def test_no_mark_of_provenance_left_in_a_caption(S, cat_specs):
    """No caption text of the catalogue page keeps a mark of provenance; each mark is on the provenance line."""
    rows, _ = S.catalogue_rows([spec_meta(s) for s in cat_specs])
    marked = 0
    for m, text, _, _ in rows:
        for k in ("operator", "grade", "params", "chip", "family", "item", "scale", "title", "search"):
            assert not re.search(r"\[module|\(module|module level|\(D4|declared (map|ladder)|estimator floor|"
                                 r"video draft|\bcited\b", text[k]), (m["clip_id"], k, text[k])
        for e in text["provenance"]:                         # the chip's meanings carry no internal code either
            assert not re.search(r"D4|rule [A-Z]\d", e["meaning"]), (m["clip_id"], e)
        marked += bool(text["provenance"])
    assert marked > 50
    wcrs = [text for m, text, _, _ in rows if m["clip_id"].startswith("post_W_writers_cramp_g")]
    assert wcrs and all(t["scale"] == "WCRS" and any(e["about"] == "WCRS (item scores)" for e in t["provenance"])
                        for t in wcrs)                       # the scale has its source; the item scores are chosen


# ---------------------------------------------------------------- wave SH1d: the uniform checks of the skeleton
def _walk_points(n=60, fps=30.0):
    """(P (n, K, 3) mm, names) of a still body: a few joints and the foot points on a floor at y = 0."""
    import numpy as np
    names = ["pelvis", "left_hip", "right_hip", "left_foot", "left_heel", "left_forefoot_lateral", "left_toe_tip",
             "left_wrist"]
    P = np.zeros((n, len(names), 3))
    P[:, 0] = [0, 900, 0]
    P[:, 1], P[:, 2] = [90, 850, 0], [-90, 850, 0]
    P[:, 3], P[:, 4], P[:, 5], P[:, 6] = [90, 18, 100], [90, 30, -60], [140, 18, 40], [90, 18, 160]
    P[:, 7] = [250, 900, 0]
    return P, names


REST_MM = {"left_foot": 18.0, "left_heel": 30.0, "left_forefoot_lateral": 18.0, "left_toe_tip": 18.0}


def test_continuity_check(S):
    import numpy as np
    P, names = _walk_points()
    r = S.check_continuity(P, names, 30.0, (), 0.0, REST_MM)
    assert r["ok"] and r["accel_max_m_s2"] == 0.0 and r["foot_jumps"] == 0
    snap = P.copy()
    snap[30, 7, 0] += 200.0                                  # a wrist out and back in one frame: 360 m/s2
    r = S.check_continuity(snap, names, 30.0)
    assert not r["ok"] and r["accel_point"] == "left_wrist" and r["accel_t_s"] == 1.0 and r["accel_frames_over"] == 3
    assert r["accel_max_m_s2"] == 360.0
    assert S.check_continuity(snap, names, 30.0, [[0.95, 1.02]])["ok"]         # inside a declared jerk
    smooth = P.copy()
    smooth[:, 7, 0] += 100.0 * np.sin(np.linspace(0, np.pi, 60))               # 100 mm over 2 s: about 1 m/s2
    assert S.check_continuity(smooth, names, 30.0)["ok"]
    jump = P.copy()
    jump[30:33, 3, 2] += 40.0                                # the planted toe joint snaps 40 mm and back
    r = S.check_continuity(jump, names, 30.0, (), 0.0, REST_MM)
    assert not r["ok"] and r["foot_jump_point"] == "left_foot" and r["foot_jump_max_mm"] == 40.0
    skim = P.copy()                                          # the toe joint skims the floor in swing at 1.2 m/s
    skim[:, 3, 2] = 100.0 + 40.0 * np.arange(60)
    assert S.check_continuity(skim, names, 30.0, (), 0.0, REST_MM)["foot_jumps"] == 0
    land = P.copy()                     # a landing step of 28 mm, a 6 mm settle, then a snap of 53 mm: the snap counts
    z = np.r_[100.0 - 28.0 * np.arange(28, 0, -1), 100.0, 106.0, np.full(30, 159.0)]
    land[:, 3, 2] = z
    land[:28, 3, 1] += 40.0                                  # in the swing above the floor before the landing
    r = S.check_continuity(land, names, 30.0, (), 0.0, REST_MM)
    assert not r["ok"] and r["foot_jumps"] == 1 and r["foot_jump_max_mm"] == 53.0 and r["foot_jump_t_s"] == 1.0
    assert S.check_continuity(land, names, 30.0, [[0.9, 1.1]], 0.0, REST_MM)["foot_jumps"] == 0
    strike = P.copy()                   # a heel strike: the landing step itself is not a snap of a planted point
    strike[:, 3, 2] = np.r_[100.0 - 40.0 * np.arange(30, 0, -1), np.full(30, 100.0)]
    strike[:29, 3, 1] += 40.0
    assert S.check_continuity(strike, names, 30.0, (), 0.0, REST_MM)["foot_jumps"] == 0


def _arm_points(n=60, fps=30.0, amp_deg=20.0, hz=3.0):
    """(P (n, 6, 3) mm, names): a still trunk and a straight left arm swinging in the sagittal plane about the
    shoulder, amp_deg at hz (the upper arm's peak angular speed 2 pi hz amp_deg)."""
    import numpy as np
    names = ["pelvis", "neck", "head", "left_shoulder", "left_elbow", "left_wrist"]
    t = np.arange(n) / fps
    th = np.radians(amp_deg) * np.sin(2 * np.pi * hz * t)
    P = np.zeros((n, 6, 3))
    P[:, 0], P[:, 1], P[:, 2], P[:, 3] = [0, 900, 0], [0, 1400, 0], [0, 1500, 0], [200, 1400, 0]
    u = np.stack([np.zeros(n), -np.cos(th), np.sin(th)], 1)
    P[:, 4] = P[:, 3] + 300.0 * u
    P[:, 5] = P[:, 3] + 550.0 * u
    return P, names


def test_continuity_rates(S):
    """PLAN M9 and M10: a segment of a symptom clip turning faster than its bound and than its level 0 fails the
    continuity check, below ACCEL_MAX_M_S2; a level 0 as fast, a slow swing and a tremor pass; a clip without a
    level 0 is read, not judged."""
    fast, names = _arm_points()                              # 20 deg at 3 Hz: about 377 deg/s, elbow 68 m/s2
    still, _ = _arm_points(amp_deg=0.0)
    r0 = S.rate_peaks(still, names)
    assert set(r0) == {"trunk", "neck", "left_upper_arm", "left_elbow"} and r0["left_upper_arm"]["deg_s"] == 0.0
    assert 0.0 < r0["trunk"]["tol_deg_s"] < r0["neck"]["tol_deg_s"]      # a shorter segment, a larger rounding
    r = S.check_continuity(fast, names, 30.0, (), None, None, r0, "invol_chorea_E7_g4")
    assert not r["ok"] and r["accel_frames_over"] == 0
    assert [x["reading"] for x in r["rates_over"]] == ["left_upper_arm"]
    assert 330.0 < r["rates_over"][0]["deg_s"] < 380.0 and r["rates_over"][0]["cap_deg_s"] == 250.0
    assert S.check_continuity(fast, names, 30.0, (), None, None, S.rate_peaks(fast, names), "x_g4")["ok"]
    slow, _ = _arm_points(hz=0.5)
    assert S.check_continuity(slow, names, 30.0, (), None, None, r0, "x_g4")["ok"]
    r = S.check_continuity(fast, names, 30.0, (), None, None, r0, "tremor_E2w_wing_tetras3p5")
    assert r["ok"] and "tremor" in r["rates_exempt"] and r["rates_over"] == []
    r = S.check_continuity(fast, names, 30.0)                 # a level 0 itself: read, not judged
    assert r["ok"] and r["rates_deg_s"]["left_upper_arm"] > 330.0
    assert S.check_continuity(fast, names, 30.0, [[0.0, 2.0]], None, None, r0, "x_g4")["ok"]   # inside an event
    assert S.rate_kind("left_upper_arm") == "upper_arm" and S.rate_kind("trunk") == "trunk"


def test_sinking_check(S):
    import numpy as np
    names = ["pelvis", "left_heel_sole", "left_lateral_sole"]
    P = np.zeros((30, 3, 3))
    P[:, 0, 1], P[:, 1, 1], P[:, 2, 1] = 900.0, 8.0, 5.0
    r = S.check_sinking(P, names, 0.0)
    assert r["ok"] and r["depth_mm"] == -5.0 and r["share"] == 0.0
    sunk = P.copy()
    sunk[10:20, 1, 1] = -8.0                                 # the heel 8 mm under the floor for a third of the clip
    r = S.check_sinking(sunk, names, 0.0)
    assert not r["ok"] and r["depth_mm"] == 8.0 and r["point"] == "left_heel_sole" and r["share"] == 0.333
    assert r["t_s"] == round(10 / 30.0, 3)
    # open-body port, wave 7: no clip of the catalogue is built on a motion-capture recording; the exemption
    # tables keep their rule for one (render_skeleton_catalogue.CAPTURED, the prefix 'mocap_'), read here on
    # ids of that form
    r = S.check_sinking(sunk, names, 0.0, clip_id="mocap_freezing_E8_g2_akinesia")   # motion capture: read only
    assert r["ok"] is None and "motion-capture" in r["exempt"] and r["depth_mm"] == 8.0
    assert S.check_sinking(sunk, names, 0.0, clip_id="gait_E8_level0")["ok"] is False
    sunk[10:20, 1, 1] = -4.0                                 # within the contact tolerance
    assert S.check_sinking(sunk, names, 0.0)["ok"]
    assert S.check_sinking(sunk, names, 0.0, supine=True)["ok"] is None
    assert S.check_sinking(P[:, :1], names[:1], 0.0)["ok"] is None


def test_skate_check(S):
    """Skate read per foot by its least-moving point (wave SH1d review of lane page): a foot that glides fails, a
    roll or a pivot (one point still) passes, a toe tip touching alone and gliding fails, a point touching for a few
    frames does not hide a glide, a skim in swing and a lifted foot are no contact."""
    import numpy as np
    P, names = _walk_points()
    foot = [names.index(k) for k in REST_MM]
    r = S.check_skate(P, names, 30.0, 0.0, REST_MM)
    assert r["ok"] and r["contacts"] == 1 and r["travel_max_mm"] == 0.0
    glide = P.copy()
    glide[:, foot, 0] += np.linspace(0.0, 12.0, 60)[:, None]  # the whole foot glides 12 mm in 2 s at contact height
    r = S.check_skate(glide, names, 30.0, 0.0, REST_MM)
    assert not r["ok"] and r["contacts_over"] == 1 and 10.0 < r["travel_max_mm"] < 12.0
    roll = P.copy()
    roll[:, 5, 0] += np.linspace(0.0, 12.0, 60)              # the lateral border moves, the heel and toes stay: a roll
    assert S.check_skate(roll, names, 30.0, 0.0, REST_MM)["ok"]
    toe = P.copy()                                           # a plantarflexed foot: only the toe tip touches, gliding
    toe[:, [3, 4, 5], 1] += 40.0
    toe[:, 6, 0] += np.linspace(0.0, 60.0, 60)
    r = S.check_skate(toe, names, 30.0, 0.0, REST_MM)
    assert not r["ok"] and r["point"] == "left_toe_tip" and r["travel_max_mm"] > 50.0
    brief = glide.copy()                                     # the heel stands still for 6 frames of a gliding contact
    brief[:, 4] = P[:, 4] + [0.0, 40.0, 0.0]
    brief[10:16, 4, 1] = 30.0
    r = S.check_skate(brief, names, 30.0, 0.0, REST_MM)
    assert not r["ok"] and r["point"] != "left_heel"
    skim = P.copy()                                          # the toe joint skims the floor in swing at 1.2 m/s
    skim[:, 3, 2] = 100.0 + 40.0 * np.arange(60)
    r = S.check_skate(skim, names, 30.0, 0.0, REST_MM)
    assert r["ok"] and r["contacts"] == 1
    short = P.copy()                                         # a contact of four frames: too short to read
    short[:, foot, 1] += 60.0
    short[20:24, 6, 1] = 18.0
    assert S.check_skate(short, names, 30.0, 0.0, REST_MM)["contacts"] == 0
    lifted = glide.copy()
    lifted[:, foot, 1] += 30.0                               # off the floor: no contact, no skate
    assert S.check_skate(lifted, names, 30.0, 0.0, REST_MM)["contacts"] == 0
    # open-body port, wave 7: no clip of the catalogue is built on a motion-capture recording; the exemption
    # tables keep their rule for one (render_skeleton_catalogue.CAPTURED, the prefix 'mocap_'), read here on
    # ids of that form
    for c in ("mocap_walk_E8", "mocap_stand_E7", "mocap_freezing_E8_g2_akinesia"):   # motion capture: read only
        r = S.check_skate(glide, names, 30.0, 0.0, REST_MM, c)
        assert r["ok"] is None and "motion-capture" in r["exempt"] and r["contacts_over"] == 1, c
    for c in ("gait_E8_level0", "ref_arise_E7", "post_E7_hemidystonia"):
        assert S.check_skate(glide, names, 30.0, 0.0, REST_MM, c)["ok"] is False, c
    # a freezing clip on the captured walk is judged inside its declared holds (wave SH1d review r3 of lane page)
    fz = "mocap_freezing_E8_g2_trembling"
    r = S.check_skate(glide, names, 30.0, 0.0, REST_MM, fz, [[0.5, 1.5]])
    assert r["ok"] is False and r["holds"]["contacts_over"] == 1 and "freezing holds" in r["exempt"]
    assert 0.5 <= r["holds"]["span_s"][0] < r["holds"]["span_s"][1] <= 1.5
    held = glide.copy()                                      # the foot glides outside the hold, stands in it
    held[15:45, foot] = held[15, foot]
    r = S.check_skate(held, names, 30.0, 0.0, REST_MM, fz, [[0.5, 1.5]])
    assert r["ok"] is True and r["holds"]["contacts"] == 1 and r["holds"]["travel_max_mm"] == 0.0
    assert r["contacts_over"] == 1                           # the captured walk outside it: read, not judged
    assert S.check_skate(glide, names, 30.0, 0.0, REST_MM, "gait_E8_level0", [[0.5, 1.5]])["ok"] is False
    tr = {"params": {"episodes": [{"onset_s": 9.0, "hold_s": [9.512, 11.75], "ramp_in_s": [9.0, 9.512]},
                                  {"onset_s": 30.0}]}}
    assert S.hold_spans(tr) == [[9.512, 11.75]] and S.hold_spans({"params": {}}) == [] and S.hold_spans(None) == []


def test_statue_check(S):
    import numpy as np
    n = 150
    P = np.zeros((n, 22, 3))
    t = np.arange(n) / 30.0
    P[:, :, 0] = 5.0 * np.sin(2 * np.pi * 2.0 * t)[:, None]   # a 10 mm sway at 2 Hz: alive
    assert S.check_statue(P)["ok"]
    still = P.copy()
    still[60:110] = still[60]                                # 1.67 s with every joint fixed
    r = S.check_statue(still)
    assert not r["ok"] and r["spans_s"] == [[2.0, 3.667]] and r["statue_s"] == 1.67
    assert S.check_statue(still, supine=True)["ok"] and S.check_statue(still[:20])["ok"]
    near = still.copy()
    near[60:110, 5, 1] += 1.0 * (np.arange(50) % 2)          # one joint moving by the export's 1 mm step: alive
    r = S.check_statue(near)
    assert r["ok"] and r["near_still_s"] == 1.67              # but still under the plan's 2 mm, read for lane physio
    assert S.check_statue(P)["near_still_s"] == 0.0 and r["near_still_s"] >= S.check_statue(still)["statue_s"]


def test_pair_floor_check(S):
    """The floor of the pair read at the soles of the lower foot: a body 5 mm higher fails, one foot plantarflexed
    (its points off the floor) and taps of the forefoot down to the floor pass."""
    import numpy as np
    n = 60
    rest = {f"{s}_{p}": h for s in ("left", "right") for p, h in (("foot", 18.0), ("heel_sole", 5.0),
                                                                  ("lateral_sole", 5.0))}
    names = list(S.SK.JOINT_NAMES[:22]) + [f"{s}_{p}" for s in ("left", "right") for p in ("heel_sole",
                                                                                           "lateral_sole")]
    ix = {nm: i for i, nm in enumerate(names)}
    lv = np.zeros((n, len(names), 3))
    lv[:, :, 1] = 500.0
    for k, h in rest.items():
        lv[:, ix[k], 1] = h
    for s in ("left", "right"):
        lv[:, ix[f"{s}_ankle"], 1] = 80.0
    up = lv + [0.0, 5.0, 0.0]                                # the symptom clip's body 5 mm higher
    r = S.check_pair_floor(up, names, lv, names, rest)
    assert r["ok"] is False and r["diff_mm"] == 5.0 and r["clip_mm"] == 5.0 and r["level0_mm"] == 0.0
    assert S.check_pair_floor(lv + [0.0, 1.0, 0.0], names, lv, names, rest)["ok"] is True
    pf = lv.copy()                                           # the left foot plantarflexed, on its toe tip
    for p, dy in (("left_foot", 5.0), ("left_heel_sole", 60.0), ("left_lateral_sole", 13.0)):
        pf[:, ix[p], 1] += dy
    r = S.check_pair_floor(pf, names, lv, names, rest)
    assert r["ok"] is True and r["diff_mm"] == 0.0 and r["foot"] == "right"
    assert r["diff_by_foot_mm"] == {"left": 5.0, "right": 0.0}
    tap = lv.copy()                                          # the right forefoot taps 4 mm into the floor's reading
    tap[::6, ix["right_lateral_sole"], 1] -= 4.0
    assert S.check_pair_floor(tap, names, lv, names, rest)["ok"] is True
    moving = lv.copy()
    moving[:, ix["left_foot"], 2] = np.linspace(0.0, 2000.0, n)          # never still
    assert S.check_pair_floor(moving, names, lv, names, rest)["ok"] is None
    held = lv.copy()                     # both feet held still 22 mm in the air for the first half, then set down
    for s in ("left", "right"):
        for p in ("ankle", "foot", "heel_sole", "lateral_sole"):
            held[:30, ix[f"{s}_{p}"], 1] += 22.0
    assert S.standing_frames(held[:, :22] / 1000.0).sum() > 50     # still in the air: standing by the foot joints
    r = S.check_pair_floor(held, names, lv, names, rest)
    assert r["ok"] is True and r["diff_mm"] == 0.0 and r["clip_mm"] == 0.0
    assert S.foot_floor_mm(held, names, rest) == {"left": 0.0, "right": 0.0}


def _sphere(radius, centre, n=400):
    """(v, n) points on a sphere and their outward normals (a Fibonacci lattice)."""
    import numpy as np
    k = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
    u = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    return np.asarray(centre) + radius * u, u


def _cylinder(radius, x0, x1, n=40, m=24):
    """(v, n) points on a cylinder about the x axis (the femur frame's axis) and their outward normals."""
    import numpy as np
    a = np.linspace(0, 2 * np.pi, m, endpoint=False)
    x = np.linspace(x0, x1, n)
    X, A = np.meshgrid(x, a)
    nrm = np.stack([np.zeros_like(A), np.cos(A), np.sin(A)], -1).reshape(-1, 3)
    return np.stack([X, radius * np.cos(A), radius * np.sin(A)], -1).reshape(-1, 3), nrm


def _contact_scene(gap_mm=50.0, finger_in_head=False):
    """(P (2, K, 3) mm, names, geo): a seated body with a head sphere and two thigh cylinders (femur frames along +z,
    the top of the thigh anterior), the palms over the thighs gap_mm above their skin."""
    import numpy as np
    hv, hn = _sphere(90.0, (0.0, 70.0, 0.0))
    geo = {"head": {"v": hv, "n": hn, "y_min": -10.0,
                    "frame_points": {"head": np.zeros(3), "head_centre": np.array([0.0, 70.0, 20.0]),
                                     "nose": np.array([0.0, 70.0, 100.0]), "left_ear": np.array([75.0, 60.0, 0.0]),
                                     "right_ear": np.array([-75.0, 60.0, 0.0])}},
           "foot_rest_mm": {}}
    for s in ("left", "right"):
        tv, tn = _cylinder(70.0, 100.0, 400.0)
        geo[f"thigh_{s}"] = {"v": tv, "n": tn, "length": 400.0}
        frame = np.array([[0, 0, 0], [-20, 0, 80], [0, 0, 85], [20, 0, 80], [40, 0, 70]], dtype=float)
        geo[f"palm_{s}"] = {"frame": frame, "v": np.array([[0.0, -15.0, 40.0], [10.0, -15.0, 60.0]])}
    names, pts = [], []

    def add(n, p):
        names.append(n)
        pts.append(np.asarray(p, dtype=float))
    head = np.array([0.0, 1300.0, 0.0])
    for k, v in geo["head"]["frame_points"].items():
        add(k, head + v)
    for s, x in (("left", 90.0), ("right", -90.0)):
        add(f"{s}_hip", [x, 500, 0])
        add(f"{s}_knee", [x, 500, 400])
        add(f"{s}_ankle", [x, 50, 400])
        add(f"{s}_foot", [x, 0, 500])
    for s, x in (("left", 90.0), ("right", -90.0)):
        wrist = np.array([x, 570.0 + 15.0 + gap_mm, 150.0])
        for j, off in zip(("wrist", "index1", "middle1", "ring1", "pinky1"), geo[f"palm_{s}"]["frame"]):
            add(f"{s}_{j}", wrist + off)
        add(f"{s}_index_tip", wrist + [0, 0, 160])
    P = np.stack(pts)[None].repeat(2, 0)
    ix = {n: i for i, n in enumerate(names)}
    if finger_in_head:
        P[1, ix["left_index_tip"]] = head + [0.0, 60.0, 30.0]
    return P, names, geo


def test_self_contact_check(S):
    import numpy as np
    P, names, geo = _contact_scene(gap_mm=50.0)
    r = S.check_self_contact(P, names, geo, "x_E1_g2")
    assert r["ok"] and r["contacts"] == [] and r["gaps"] == []
    r = S.check_self_contact(P, names, geo, "invol_E1_level0")            # 'hands resting on the thighs'
    assert not r["ok"] and [(g["hand"], g["gap_mm"]) for g in r["gaps"]] == [("left", 50.0), ("right", 50.0)]
    assert r["touches"] == ["hands resting on the thighs"]
    P2, names, geo = _contact_scene(gap_mm=5.0)
    assert S.check_self_contact(P2, names, geo, "invol_E1_level0")["ok"]
    assert S.check_self_contact(P2, names, geo, "invol_chorea_E1_g2", "invol_E1_level0")["gaps"] == []
    P3, names, geo = _contact_scene(gap_mm=-40.0)                           # the palm 40 mm into the thigh
    r = S.check_self_contact(P3, names, geo, "x_E1_g2")
    assert not r["ok"] and {c["with"] for c in r["contacts"]} == {"thigh_left", "thigh_right"}
    P4, names, geo = _contact_scene(finger_in_head=True)
    r = S.check_self_contact(P4, names, geo, "x_E1_g2")
    assert [c["with"] for c in r["contacts"]] == ["head"] and r["contacts"][0]["point"] == "left_index_tip"
    assert r["contacts"][0]["t_s"] == round(1 / 30.0, 3)
    P5, names, geo = _contact_scene(gap_mm=5.0)
    ix = {n: i for i, n in enumerate(names)}
    right = [i for n, i in ix.items() if n.startswith("right_") and not n.endswith(("hip", "knee", "ankle", "foot"))]
    P5[:, right] += P5[0, ix["left_wrist"]] - P5[0, ix["right_wrist"]] + [0.0, 0.0, 8.0]   # the hands 8 mm apart
    r = S.check_self_contact(P5, names, geo, "x_E1_g2")
    assert any(c["with"] == "hands" and c["mm"] == 8.0 for c in r["contacts"])
    hands = [c for c in r["contacts"] if c["with"] == "hands"][0]      # where the hands meet, by point
    assert hands["point"].startswith("left_") and " and right_" in hands["point"]
    r = S.check_self_contact(P5, names, geo, "park_E21_applause_g2")       # 'meeting palm to palm': allowed
    assert not any(c["with"] == "hands" for c in r["contacts"])
    P6, names, geo = _contact_scene(gap_mm=50.0)              # two fingertips 9.8 mm apart, the hands apart
    P6[:, ix["right_index_tip"]] = P6[:, ix["left_index_tip"]] + [0.0, 0.0, 9.8]
    assert not any(c["with"] == "hands" for c in S.check_self_contact(P6, names, geo, "x_E1_g2")["contacts"])
    P6[:, ix["right_index_tip"]] = P6[:, ix["left_index_tip"]] + [0.0, 0.0, 3.0]    # overlapping tips
    r = S.check_self_contact(P6, names, geo, "x_E1_g2")
    assert [(c["mm"], c["point"]) for c in r["contacts"] if c["with"] == "hands"] == [
        (3.0, "left_index_tip and right_index_tip")]
    P6[:, ix["right_index_tip"]] = P6[:, ix["left_index1"]] + [0.0, 0.0, 12.0]      # a tip at a joint centre
    r = S.check_self_contact(P6, names, geo, "x_E1_g2")
    assert [(c["mm"], c["point"]) for c in r["contacts"] if c["with"] == "hands"] == [
        (12.0, "left_index1 and right_index_tip")]
    assert S.check_self_contact(_contact_scene(gap_mm=-6.0)[0], names, geo, "x_E1_g2")["ok"]   # within THIGH_TOL_MM
    assert not S.check_self_contact(_contact_scene(gap_mm=-10.0)[0], names, geo, "x_E1_g2")["ok"]
    names_t = names + ["pelvis", "neck"]                     # the trunk: hip flexion 90 deg, then 130 deg
    seated = np.concatenate([P, np.array([[0.0, 500.0, 0.0], [0.0, 1100.0, 0.0]])[None].repeat(2, 0)], 1)
    assert np.allclose(S.hip_flexion_deg(seated, {k: i for i, k in enumerate(names_t)}, "left"), 90.0)
    r = S.check_self_contact(seated, names_t, geo, "invol_E1_level0")
    assert len(r["gaps"]) == 2 and "flexed 60 to 110 deg" in r["gaps"][0]["reading"]
    folded = seated.copy()
    a = np.radians(130.0)
    folded[:, -1] = [0.0, 500.0 + 600.0 * np.sin(a), -600.0 * np.cos(a)]
    assert np.allclose(S.hip_flexion_deg(folded, {k: i for i, k in enumerate(names_t)}, "left"), 130.0)
    r = S.check_self_contact(folded, names_t, geo, "invol_E1_level0")   # the rigid thigh is off at deep flexion
    assert r["ok"] and r["gaps"] == [] and [g["hand"] for g in r["gaps_not_read"]] == ["left", "right"]
    # open-body port, wave 7: no clip of the catalogue is built on a motion-capture recording; the exemption
    # tables keep their rule for one (render_skeleton_catalogue.CAPTURED, the prefix 'mocap_'), read here on
    # ids of that form
    r = S.check_self_contact(P3, names, geo, "mocap_walk_E8")           # motion capture: thighs read, not judged
    assert r["ok"] and r["contacts"] == [] and "motion-capture" in r["exempt"]
    assert {c["with"] for c in r["contacts_exempt"]} == {"thigh_left", "thigh_right"}
    assert not S.check_self_contact(P4, names, geo, "mocap_freezing_E8_g2_akinesia")["ok"]   # the head still judged
    assert S.declared_touches("invol_chorea_E1_g2", "invol_E1_level0")[0][4] is False
    assert [t[2] for t in S.declared_touches("post_E7_cd_geste_C1")] == ["allowed"]
    assert np.isfinite(S.signed_distance(np.zeros((1, 3)), geo["head"])).all()


def test_a_reach_touch_its_module_reads_on_the_mesh_is_judged_on_that_reading(S):
    """Plan phase E (MESH_TOUCHES): Gowers' climbing hands read 27 mm off the rigid thigh stand-in at the deep flexion
    where the module's mesh puts them 3 to 6 mm from the skin; a declared 'reach' touch that the clip's truth reads on
    the mesh is judged on that reading against the same TOUCH_GAP_MM, the rigid reading kept beside it."""
    import numpy as np
    P, names, geo = _contact_scene(gap_mm=50.0)
    names_t = names + ["pelvis", "neck"]
    seated = np.concatenate([P, np.array([[0.0, 500.0, 0.0], [0.0, 1100.0, 0.0]])[None].repeat(2, 0)], 1)
    r = S.check_self_contact(seated, names_t, geo, "park_E7_gowers")       # the rigid thigh: 50 mm, a gap
    assert [(g["hand"], g["gap_mm"]) for g in r["gaps"]] == [("left", 50.0), ("right", 50.0)]
    truth = {"validity": {"hand_thigh": {"ok": True, "touch_mm": {"left": 3.004, "right": 5.561}}}}
    touches = S.touch_readings(truth)
    assert touches == {"thighs": {"left": 3.0, "right": 5.56}} and S.touch_readings({"validity": {}}) == {}
    r = S.check_self_contact(seated, names_t, geo, "park_E7_gowers", fps=S.FPS, mesh_touches=touches)
    assert r["ok"] and r["gaps"] == []
    assert [(t["hand"], t["mesh_gap_mm"], t["rigid_gap_mm"]) for t in r["touches_on_mesh"]] == [
        ("left", 3.0, 50.0), ("right", 5.56, 50.0)]
    far = {"thighs": {"left": S.TOUCH_GAP_MM + 1.0, "right": 5.0}}             # the mesh reading still judged
    r = S.check_self_contact(seated, names_t, geo, "park_E7_gowers", fps=S.FPS, mesh_touches=far)
    assert not r["ok"] and [(g["hand"], g["gap_mm"]) for g in r["gaps"]] == [("left", S.TOUCH_GAP_MM + 1.0)]
    r = S.check_self_contact(seated, names_t, geo, "invol_E1_level0", fps=S.FPS, mesh_touches=touches)
    assert len(r["gaps"]) == 2 and "touches_on_mesh" not in r                  # a 'rest' touch: the rigid reading


@needs_body
def test_contact_geometry_of_the_rest_mesh(S):
    """The surfaces read on the neutral rest mesh: the head centre inside the head, a point in front of the face
    outside; the femur's mid-point inside the thigh; the palm's skin under the knuckles; the foot points' heights
    above the sole (the toe-block joint at viewer.sole_offset_m)."""
    import numpy as np
    g = S.contact_geometry()
    hc = 1000.0 * np.asarray(S.SK.head_landmarks()["head_centre"])
    # inside the head: 30 mm deep on SMPL+H; on the open body 20 mm (Anny's mouth surface runs back into the head,
    # its centre the extent's midpoint, skeleton.head_landmarks; open-body port wave 1)
    from mdfm.synth.body import is_open
    assert S.signed_distance(hc[None], g["head"])[0] < (-20.0 if is_open() else -30.0)
    assert S.signed_distance(np.array([[0.0, 70.0, 300.0]]), g["head"])[0] > 150.0
    # the rim at the base of the neck, under the head joint (on the open body lower: Anny's head joint sits at the
    # skull base and its three neck bones are the neck joint's)
    assert (-80.0 if is_open() else -60.0) < g["head"]["y_min"] < -20.0
    for s in ("left", "right"):
        L = g[f"thigh_{s}"]["length"]
        assert 300.0 < L < 450.0 and S.signed_distance(np.array([[0.6 * L, 0.0, 0.0]]), g[f"thigh_{s}"])[0] < -40.0
        pv, fr = g[f"palm_{s}"]["v"], g[f"palm_{s}"]["frame"]
        assert len(pv) > 20 and (pv[:, 1] < fr[2, 1]).all()  # under the knuckles (rest palms down)
    assert abs(g["foot_rest_mm"]["left_foot"] - 1000.0 * S.V.sole_offset_m()) < 1e-9
    assert set(g["foot_rest_mm"]) == {p for s in ("left", "right") for p in S.FOOT_POINTS[s]}
    assert set(g["toe_rest_mm"]) == {"left_toe_tip", "right_toe_tip"}
    assert all(0.0 < h < 30.0 for h in g["toe_rest_mm"].values())
    assert S.skate_rest_mm(g) == {**g["foot_rest_mm"], **g["toe_rest_mm"]}
    for s in ("left", "right"):                              # the sole points lie on the sole, under their landmarks
        for k in ("heel_sole", "lateral_sole"):
            assert 0.0 <= g["foot_rest_mm"][f"{s}_{k}"] < 12.0, (s, k)
        assert set(g[f"foot_{s}"]["frame"]) == set(S.FOOT_FRAME[s])


def test_sole_points_follow_the_foot(S):
    """The sole points are carried by the foot's pose on its four frame points (a turn of the foot carries them)."""
    import numpy as np
    fr = {"left_ankle": np.array([90.0, 74.0, -27.0]), "left_foot": np.array([115.0, 18.0, 92.0]),
          "left_heel": np.array([90.0, 34.0, -87.0]), "left_forefoot_lateral": np.array([158.0, 19.0, 78.0])}
    geo = {"foot_left": {"frame": fr, "sole": {"left_heel_sole": np.array([88.0, 8.0, -56.0])}}}
    a = np.radians(30.0)
    Rm = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    names = list(fr) + ["pelvis"]
    P = np.stack([np.stack([fr[k] for k in fr] + [np.zeros(3)]),
                  np.stack([Rm @ fr[k] + [10.0, 0.0, 5.0] for k in fr] + [np.zeros(3)])])
    P2, n2 = S.with_sole_points(P, names, geo)
    assert n2 == names + ["left_heel_sole"] and P2.shape == (2, 6, 3)
    assert np.allclose(P2[0, 5], [88.0, 8.0, -56.0]) and np.allclose(P2[1, 5], Rm @ [88.0, 8.0, -56.0] + [10, 0, 5])
    assert S.with_sole_points(P[:, 4:], names[4:], geo)[1] == ["pelvis"]          # no foot: unchanged


def test_the_sole_behind_the_toe_joint_stays_with_the_foot(S):
    """Plan phase E: the toe block's sole points behind the toe-block joint (the skin under the metatarsal heads) are
    carried by the foot's pose ('cloud_toe_bend' leaves them out of the toes' bend), those ahead of it turn with the
    toes: a still foot whose toes extend leaves its ball where it stands (the moving toes read 7.99 mm of skate on
    the ball's skin turned about the joint 25.7 mm above the sole); on the rest geometry the mask is a part of the
    toe block's, with no point behind the joint."""
    import numpy as np
    fr = {"left_ankle": np.array([0.0, 80.0, 0.0]), "left_foot": np.array([0.0, 26.0, 120.0]),
          "left_heel": np.array([0.0, 40.0, -60.0]), "left_forefoot_lateral": np.array([40.0, 20.0, 110.0])}
    cloud = np.array([[0.0, 0.0, -50.0], [0.0, 0.0, 100.0], [0.0, 0.0, 150.0]])   # heel, ball (behind), toe pad
    tip0 = np.array([0.0, 20.0, 170.0])
    g = {"frame": fr, "cloud": cloud, "cloud_w": np.eye(3), "cloud_toe": np.array([False, True, True]),
         "cloud_toe_bend": np.array([False, False, True]), "toe_joint": fr["left_foot"], "toe_tip": tip0}
    a = np.radians(30.0)                                   # the toes extended by 30 deg about the joint
    Rx = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
    names = list(fr) + ["left_toe_tip"]
    tips = [tip0, fr["left_foot"] + Rx @ (tip0 - fr["left_foot"])]
    P = np.stack([np.stack([fr[k] for k in fr] + [t]) for t in tips])
    ix = {n: i for i, n in enumerate(names)}
    X = S.sole_cloud(P, ix, {"foot_left": g}, "left")
    assert np.allclose(X[1, :2], X[0, :2])                 # the heel and the ball stay
    assert np.linalg.norm(X[1, 2] - X[0, 2]) > 10.0        # the toe pad turns with the toes
    old = dict(g, cloud_toe_bend=g["cloud_toe"])           # the rule before: the ball turned with them too
    Xo = S.sole_cloud(P, ix, {"foot_left": old}, "left")
    assert np.linalg.norm((Xo[1, 1] - Xo[0, 1])[[0, 2]]) > 3.0
    geo = S.contact_geometry()
    for side in S.SIDES:
        f = geo[f"foot_{side}"]
        bend, toe = f["cloud_toe_bend"], f["cloud_toe"]
        assert (bend <= toe).all() and bend.sum() < toe.sum()
        assert (f["cloud"][bend, 2] >= f["toe_joint"][2]).all()


def test_checks_refuse(S, monkeypatch):
    """Every check refuses since 2026-11-11 (SKELETON_REPORTED_ONLY empty): the page selection hides a clip a check
    fails (SKELETON_FAIL and the check), the clips drawn beside it as their level 0 and a level 0 it leaves alone
    (refusal_followers), keeps a clip read and not judged (ok None), and hands on the checks of the clips shown."""
    assert S.SKELETON_REPORTED_ONLY == ()
    ms = [_fake_meta("a_E1_level0", grade=0), _fake_meta("a_E1_x_g2", level0="a_E1_level0"),
          _fake_meta("b_E1_level0", grade=0), _fake_meta("b_E1_x_g2", level0="b_E1_level0"),
          _fake_meta("c_E1_level0", grade=0), _fake_meta("c_E1_x_g2", level0="c_E1_level0")]
    specs = []
    for m in ms:
        s = type("Spec", (), {})()
        s.clip_id, s.level0 = m["clip_id"], m["level0"]
        specs.append(s)
    res = {m["clip_id"]: {k: {"ok": True} for k in S.SKELETON_CHECKS} for m in ms}
    res["a_E1_level0"]["statue"] = {"ok": False}
    res["b_E1_x_g2"]["skate"] = {"ok": False}
    res["c_E1_x_g2"]["skate"] = {"ok": None, "exempt": "read, not judged"}
    monkeypatch.setattr(S, "hidden_clips", lambda metas, blobs, specs: {})
    monkeypatch.setattr(S, "skeleton_checks", lambda metas, blobs: {m["clip_id"]: res[m["clip_id"]] for m in metas})
    metas, blobs, hidden, checks = S.page_selection(ms, {m["clip_id"]: b"" for m in ms}, specs)
    assert hidden == {"a_E1_level0": [f"{S.SKELETON_FAIL}statue"], "a_E1_x_g2": [S.LEVEL0_REFUSED],
                      "b_E1_x_g2": [f"{S.SKELETON_FAIL}skate"], "b_E1_level0": [S.LONE_LEVEL0]}
    assert [m["clip_id"] for m in metas] == ["c_E1_level0", "c_E1_x_g2"] and set(blobs) == set(checks)
    assert set(checks) == {"c_E1_level0", "c_E1_x_g2"}


def test_checks_failed_and_page_sentence(S):
    """checks_failed names the checks a clip fails (ok False), the manifest block keeps their readings, and the page
    names a clip a check refuses in its own sentence."""
    assert S.SKELETON_CHECKS == ("continuity", "self_contact", "skate", "statue", "pair_floor", "sinking")
    res = {"a_g2": {"continuity": {"ok": False}, "statue": {"ok": True}, "pair_floor": {"ok": None}},
           "b_g2": {"continuity": {"ok": True}, "statue": {"ok": False}}}
    assert S.checks_failed(res) == {"a_g2": ["continuity"], "b_g2": ["statue"]}
    assert S.checks_failed(res, ("statue",)) == {"b_g2": ["statue"]}
    rec = S.checks_record(res)
    assert rec["skeleton_checks"] == {"a_g2": {"continuity": {"ok": False}}, "b_g2": {"statue": {"ok": False}}}
    assert rec["skeleton_checks_flagged"]["continuity"] == 1 and rec["skeleton_checks_read"] == 2
    t = S.catalogue_template({"a_g2": [f"{S.SKELETON_FAIL}continuity"]}, {"a_g2": "clip a"})
    assert ("1 clip whose skeleton fails the page's own checks (continuity, contact between body parts, foot skate, "
            "a body frozen for a second, the floor of the pair, a foot sinking into the floor) is not shown: "
            "clip a.") in t
    assert "No clip failed the validity checks" in t


@needs_body
def test_checks_on_a_built_pair(S, tmp_path):
    """The checks of two built clips run from their data files: a seated level 0 and its rest tremor read clean
    for the floor of the pair, the clip drawn on the level 0's floor."""
    from mdfm.synth import exam as EX
    specs = {s.clip_id: s for s in EX.exam_set()}
    out = str(tmp_path / "cat").replace("\\", "/")
    sel = [specs["tremor_E1_rest_g0"], specs["tremor_E1_rest_g2"]]
    st = S.CodeState(sel, [])
    for s in sel:
        assert S.export_clip(s, out, S.SET_CATALOGUE, module_checks=True, code=st.of(s))[1] is None
    metas, blobs = S.load_all(out, sel)
    assert all(m["floor"]["standing_frames"] > 1000 and "still" in m["floor"]["rule"] for m in metas)
    assert all(m["events_s"] == [] and m["slides_s"] == {} for m in metas)
    res = S.skeleton_checks(metas, blobs)
    assert set(res) == {"tremor_E1_rest_g0", "tremor_E1_rest_g2"}
    assert set(res["tremor_E1_rest_g2"]) == set(S.SKELETON_CHECKS) and "pair_floor" not in res["tremor_E1_rest_g0"]
    assert res["tremor_E1_rest_g2"]["pair_floor"]["ok"] is True
    assert res["tremor_E1_rest_g2"]["skate"]["ok"] and res["tremor_E1_rest_g2"]["continuity"]["accel_frames_over"] == 0
    old = [{k: v for k, v in m.items() if k not in ("events_s", "slides_s")} for m in metas]
    r_old = S.skeleton_checks(old, blobs)["tremor_E1_rest_g2"]   # data exported before the declared spans
    assert r_old["continuity"]["ok"] is None and "events_s" in r_old["continuity"]["not_judged"]
    assert r_old["skate"]["ok"] is None and r_old["statue"] == res["tremor_E1_rest_g2"]["statue"]
    page, nbytes, _, notes = S.build_page(out, metas, blobs, [], mode="catalogue", set_name=S.SET_CATALOGUE)
    _, man = S.write_manifest(out, metas, blobs, page, nbytes, [], {}, notes, set_name=S.SET_CATALOGUE, specs=sel,
                              modules=[], stale={}, checks=res)
    assert man["skeleton_checks_read"] == 2 and set(man["skeleton_checks_flagged"]) == set(S.SKELETON_CHECKS)
    assert man["page_floors"]["tremor_E1_rest_g2"]["root"] == "tremor_E1_rest_g0"
    with open(page, encoding="utf-8") as f:
        idx = _index(f.read())
    c = {x["id"]: x for x in idx["clips"]}
    assert c["tremor_E1_rest_g2"]["floor"] == c["tremor_E1_rest_g0"]["floor"]
    assert c["tremor_E1_rest_g2"]["boxes"] == c["tremor_E1_rest_g0"]["boxes"]


def test_declared_slides_left_out(S):
    """A dragged foot's declared slides (gait.py drag_sides: the foot slides over each of its swings, a deliberate
    skate) are no contact for the skate check and no snap for the continuity check; outside them the same glide and
    the same snap still fail (wave SH1d review of lane page, its major 1)."""
    import numpy as np
    st = np.ones((300, 2), dtype=bool)
    st[100:200, 0] = False                                  # the left foot in swing from 1 s to 2 s (100 Hz)
    st[150:160, 1] = False
    tr = {"params": {"gait_params": {"drag_sides": ("left",)}}, "arrays": {"stance": st}}
    assert S.slide_spans(tr, 100.0) == {"left": [[1.0, 2.0]]}
    assert S.slide_spans({"params": {"gait_params": {"drag_sides": ()}}, "arrays": {"stance": st}}, 100.0) == {}
    assert S.slide_spans({"params": {}}, 100.0) == {} and S.slide_spans(None, 100.0) == {}
    m = S.slide_mask(90, 30.0, {"left": [[1.0, 2.0]]}, "left")
    assert m[29] and m[30] and m[60] and m[61] and not m[28] and not m[62]           # widened by one frame
    assert not S.slide_mask(90, 30.0, {"left": [[1.0, 2.0]]}, "right").any()
    P, names = _walk_points(90)
    foot = [names.index(k) for k in REST_MM]
    drag = P.copy()                                          # the whole foot slides 120 mm from 1 s to 2 s
    drag[30:61, foot, 0] += np.linspace(0.0, 120.0, 31)[:, None]
    drag[61:, foot, 0] += 120.0
    r = S.check_skate(drag, names, 30.0, 0.0, REST_MM)
    assert r["ok"] is False and r["slide_frames_left_out"] == 0
    r = S.check_skate(drag, names, 30.0, 0.0, REST_MM, "gait_E8_functional_dragging_left", (),
                      {"left": [[1.0, 2.0]]})
    assert r["ok"] is True and r["contacts"] == 2 and r["slide_frames_left_out"] == 33
    assert S.check_skate(drag, names, 30.0, 0.0, REST_MM, "x", (), {"right": [[1.0, 2.0]]})["ok"] is False
    jump = P.copy()
    jump[45:, 3, 2] += 40.0                                  # the planted toe joint snaps 40 mm at 1.5 s
    assert S.check_continuity(jump, names, 30.0, (), 0.0, REST_MM)["foot_jumps"] == 1
    r = S.check_continuity(jump, names, 30.0, (), 0.0, REST_MM, None, "x", {"left": [[1.0, 2.0]]})
    assert r["ok"] and r["foot_jumps"] == 0


def test_skate_turning_contacts_counted_apart(S):
    """The contacts in which the foot turns by TURN_YAW_DEG at least are counted apart ('turning'), the others
    'straight'; without the sole cloud both are judged on SKATE_MM (the turn's tolerance comes with the cloud's
    covering radius, test_skate_sole_cloud)."""
    import numpy as np
    P, names = _walk_points()
    foot = [names.index(k) for k in REST_MM]
    toe = names.index("left_toe_tip")
    piv = P.copy()                                           # the foot turns 40 deg about its toe tip, which slides
    a = np.radians(np.linspace(0.0, 40.0, 60))
    for i, k in enumerate(foot):
        d = P[0, k, [0, 2]] - P[0, toe, [0, 2]]
        piv[:, k, 0] = P[0, toe, 0] + d[0] * np.cos(a) - d[1] * np.sin(a)
        piv[:, k, 2] = P[0, toe, 2] + d[0] * np.sin(a) + d[1] * np.cos(a)
    piv[:, foot, 0] += np.linspace(0.0, 10.0, 60)[:, None]   # 10 mm of slip at the pivot
    r = S.check_skate(piv, names, 30.0, 0.0, REST_MM)
    assert r["ok"] is False and r["turning"] == {"contacts": 1, "contacts_over": 1, "travel_max_mm": r["travel_max_mm"]}
    assert r["straight"]["contacts"] == 0 and 9.0 < r["travel_max_mm"] < 11.0
    glide = P.copy()
    glide[:, foot, 0] += np.linspace(0.0, 12.0, 60)[:, None]
    r = S.check_skate(glide, names, 30.0, 0.0, REST_MM)
    assert r["straight"]["contacts_over"] == 1 and r["turning"]["contacts"] == 0


FOOT_REST = {"left_ankle": (90.0, 80.0, 0.0), "left_foot": (90.0, 18.0, 120.0), "left_heel": (90.0, 30.0, -60.0),
             "left_forefoot_lateral": (140.0, 18.0, 60.0), "left_toe_tip": (90.0, 18.0, 160.0)}


def _sole_geo(cover=1.0):
    """A contact geometry of the left foot (contact_geometry's keys): its frame points (FOOT_REST), a sole cloud of
    15 points on the floor (x 60 to 120 mm, z -60 to 100 mm), sampled at its vertices, and a covering radius."""
    import numpy as np
    cloud = np.array([[x, 0.0, z] for x in (60.0, 90.0, 120.0) for z in (-60.0, -20.0, 20.0, 60.0, 100.0)])
    g = {"frame": {k: np.array(FOOT_REST[k]) for k in ("left_ankle", "left_foot", "left_heel",
                                                     "left_forefoot_lateral")},
         "sole": {}, "cloud": cloud, "cloud_w": np.eye(len(cloud)), "cloud_rest_mm": np.zeros(len(cloud)),
         "cloud_toe": np.zeros(len(cloud), dtype=bool), "cloud_region": ["ball"] * len(cloud),
         "toe_joint": np.array(FOOT_REST["left_foot"]), "toe_tip": np.array(FOOT_REST["left_toe_tip"])}
    return {"foot_left": g, "cover_mm": cover, "foot_rest_mm": {"left_foot": 18.0},
            "toe_rest_mm": {"left_toe_tip": 18.0}}


def _turned_foot(deg, about=(90.0, 60.0), slip_mm=0.0, n=60):
    """(P (n, 6, 3) mm, names): the left foot's frame points and toe tip turning by deg about the vertical through
    the floor point about (x, z) over n frames, that point slipping slip_mm along x meanwhile."""
    import numpy as np
    names = ["pelvis"] + list(FOOT_REST)
    a = np.radians(np.linspace(0.0, deg, n))
    c = np.array(about)
    P = np.zeros((n, len(names), 3))
    P[:, 0] = [90.0, 900.0, 0.0]
    for i, k in enumerate(FOOT_REST, 1):
        x, y, z = FOOT_REST[k]
        dx, dz = x - c[0], z - c[1]
        P[:, i, 0] = c[0] + dx * np.cos(a) - dz * np.sin(a) + np.linspace(0.0, slip_mm, n)
        P[:, i, 1] = y
        P[:, i, 2] = c[1] + dx * np.sin(a) + dz * np.cos(a)
    return P, names


def test_skate_sole_cloud(S):
    """The skate tolerance for turning contacts (2026-11-11): a foot pivoting about a point of its sole (the ball,
    no named point) passes with the sole cloud and fails on the named points alone; a foot that glides, or slips at
    its pivot, fails; a point standing within CONTACT_TOL_MM of its rest height but above the foot's touching patch
    (SKATE_TOUCH_MM) carries nothing; the tolerance is SKATE_MM plus the covering radius times the turn."""
    rest = {"left_foot": 18.0, "left_toe_tip": 18.0}
    geo = _sole_geo(cover=1.0)
    piv, names = _turned_foot(60.0)
    r = S.check_skate(piv, names, 30.0, 0.0, rest, geo=geo)
    assert r["ok"] is True and r["contacts"] == 1 and r["turning"]["contacts"] == 1
    assert r["travel_max_mm"] == 0.0 and r["point"] == "left_sole_ball"
    assert 3.9 < r["tol_mm"] < 4.1                            # 3 mm + 1 mm per radian over about 58 deg of turn
    r = S.check_skate(piv, names, 30.0, 0.0, rest)            # the named points alone circle the ball
    assert r["ok"] is False and r["travel_max_mm"] > 30.0 and r["tol_mm"] == S.SKATE_MM
    slip, _ = _turned_foot(60.0, slip_mm=10.0)                # 10 mm of slip at the pivot
    r = S.check_skate(slip, names, 30.0, 0.0, rest, geo=geo)
    assert r["ok"] is False and 9.0 < r["travel_max_mm"] < 10.5 and r["turning"]["contacts_over"] == 1
    glide, _ = _turned_foot(0.0, slip_mm=10.0)                # a glide without a turn
    r = S.check_skate(glide, names, 30.0, 0.0, rest, geo=geo)
    assert r["ok"] is False and r["straight"]["contacts_over"] == 1 and r["tol_mm"] == S.SKATE_MM
    still, _ = _turned_foot(0.0)
    assert S.check_skate(still, names, 30.0, 0.0, rest, geo=geo)["travel_max_mm"] == 0.0
    tip = glide.copy()                                        # the toe tip held still 3 mm above the floor
    tip[:, names.index("left_toe_tip")] = [90.0, 3.0, 160.0]
    assert S.check_skate(tip, names, 30.0, 0.0, rest)["ok"] is True          # standing (on the named points alone)
    r = S.check_skate(tip, names, 30.0, 0.0, rest, geo=geo)                  # above the touching patch: no carrier
    assert r["ok"] is False and r["point"] != "left_toe_tip"
    wide = S.check_skate(slip, names, 30.0, 0.0, rest, geo=_sole_geo(cover=8.0))   # a coarse cloud: 3 + 8 x 1 rad
    assert wide["ok"] is True and wide["tol_mm"] > 10.5
    rows = S.skate_contact_list(piv, names, 30.0, 0.0, geo, {"left": [[0.5, 1.0]]})
    assert [x["kind"] for x in rows] == ["slide"] and rows[0]["named_travel_mm"] > 30.0
    assert S.skate_contact_list(piv, names, 30.0, 0.0, geo)[0]["kind"] == "turning"


def test_sole_cloud_toes_bend(S):
    """The toe block's sole points turn about the toe-block joint with the toes (the rest toe tip direction turned
    onto the clip's), the ankle's ride the foot's pose."""
    import numpy as np
    geo = _sole_geo()
    g = geo["foot_left"]
    g["cloud_toe"] = g["cloud"][:, 2] > 80.0                  # the points at z 100 mm on the toes
    P, names = _turned_foot(0.0, n=3)
    ix = {k: i for i, k in enumerate(names)}
    P[1, ix["left_toe_tip"]] = [90.0, 18.0 + 40.0, 120.0 + 0.0]   # the toes bent up by 90 deg about the joint
    C = S.sole_cloud(P, ix, geo, "left")
    assert C.shape == (3, 15, 3) and np.allclose(C[0], g["cloud"]) and np.allclose(C[2], g["cloud"])
    toe = np.flatnonzero(g["cloud_toe"])
    assert np.allclose(C[1, ~g["cloud_toe"]], g["cloud"][~g["cloud_toe"]])
    assert np.allclose(C[1, toe, 2], 120.0 + 18.0) and np.allclose(C[1, toe, 1], 18.0 - 20.0)   # (y, z) -> (z, -y)
    assert S.sole_cloud(P[:, :3], {k: ix[k] for k in names[:3]}, geo, "left") is None          # no heel, no forefoot
    assert S.sole_cloud(P, ix, {}, "left") is None


def _trunk_points(n=60, fps=30.0, trunk_deg_s=90.0, neck_deg_s=0.0):
    """(P (n, 7, 3) mm, names): a trunk pitching forward about the pelvis at trunk_deg_s, the head on the neck
    pitching forward on the trunk at neck_deg_s, the arms hanging rigid on the trunk."""
    import numpy as np
    names = ["pelvis", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
    t = np.arange(n) / fps
    a, b = np.radians(trunk_deg_s) * t, np.radians(neck_deg_s) * t

    def rot(v, ang):                                         # about the x axis (the shoulder line): y to z
        return np.stack([np.full_like(ang, v[0]), v[1] * np.cos(ang) - v[2] * np.sin(ang),
                         v[1] * np.sin(ang) + v[2] * np.cos(ang)], 1)
    P = np.zeros((n, 7, 3))
    P[:, 0] = [0, 900, 0]
    P[:, 1] = P[:, 0] + rot([0, 500, 0], a)
    P[:, 2] = P[:, 1] + rot([0, 200, 0], a + b)
    P[:, 3], P[:, 4] = P[:, 0] + rot([180, 480, 0], a), P[:, 0] + rot([-180, 480, 0], a)
    P[:, 5], P[:, 6] = P[:, 3] + rot([0, -300, 0], a), P[:, 4] + rot([0, -300, 0], a)
    return P, names


def test_rates_of_the_neck_and_arms_on_the_trunk(S):
    """The neck and the upper arms are read in the trunk's frame (minor 3 of the wave SH1d review of lane page): a
    head and arms carried rigidly by a trunk turning at 90 deg/s read still; a neck turning on that trunk at 150
    deg/s reads 150, not the 240 of the room, under its 200 deg/s bound; without the shoulders the room is read."""
    carried, names = _trunk_points()
    r = S.rate_peaks(carried, names)
    assert 85.0 < r["trunk"]["deg_s"] < 95.0 and r["trunk"]["frame"] == "room"
    assert r["neck"]["frame"] == "trunk" and r["neck"]["deg_s"] <= r["neck"]["tol_deg_s"]
    assert r["left_upper_arm"]["deg_s"] <= r["left_upper_arm"]["tol_deg_s"]
    nod, _ = _trunk_points(neck_deg_s=150.0)
    r = S.rate_peaks(nod, names)
    assert 145.0 < r["neck"]["deg_s"] < 155.0
    still, _ = _trunk_points(trunk_deg_s=0.0)
    assert S.check_continuity(nod, names, 30.0, (), None, None, S.rate_peaks(still, names), "x_g3")["ok"]
    room = S.rate_peaks(nod[:, [0, 1, 2, 3, 5]], [names[i] for i in (0, 1, 2, 3, 5)])
    assert room["neck"]["frame"] == "room" and 235.0 < room["neck"]["deg_s"] < 245.0


def test_refusal_followers(S):
    """A clip whose level 0 a check refuses is left out with it (LEVEL0_REFUSED), and so is a level 0 whose symptom
    clips the checks refuse (LONE_LEVEL0), through a chain of level 0s; the page names them (minor 1 of the wave
    SH1d review of lane page)."""
    ms = [_fake_meta("a_E1_level0", grade=0), _fake_meta("a_E1_x_g2", level0="a_E1_level0"),
          _fake_meta("a_E1_y_g0", grade=0, level0="a_E1_level0"), _fake_meta("a_E1_y_g3", level0="a_E1_y_g0"),
          _fake_meta("b_E1_level0", grade=0), _fake_meta("b_E1_x_g2", level0="b_E1_level0"),
          _fake_meta("c_E1_level0", grade=0), _fake_meta("c_E1_x_g2", level0="c_E1_level0")]
    spec = type("Spec", (), {})
    specs = []
    for m in ms:
        s = spec()
        s.clip_id, s.level0 = m["clip_id"], m["level0"]
        specs.append(s)
    fail = [f"{S.SKELETON_FAIL}skate"]
    assert S.refusal_followers(ms, specs, {}) == {}
    out = S.refusal_followers(ms, specs, {"a_E1_level0": fail, "b_E1_x_g2": fail})
    assert out == {"a_E1_x_g2": [S.LEVEL0_REFUSED], "a_E1_y_g0": [S.LEVEL0_REFUSED],
                   "a_E1_y_g3": [S.LEVEL0_REFUSED], "b_E1_level0": [S.LONE_LEVEL0]}
    t = S.catalogue_template({"a_E1_level0": fail, "a_E1_x_g2": [S.LEVEL0_REFUSED], "b_E1_level0": [S.LONE_LEVEL0]},
                             {"a_E1_x_g2": "clip ax"})
    assert "1 clip whose level 0 fails the page's own checks is not shown either: clip ax." in t
    assert "No clip failed the validity checks" in t


def test_checks_record_keeps_what_it_does_not_judge(S):
    """The manifest keeps the readings of the clips a table of exemptions names and the ids of the clips whose data
    hold no declared spans (major 2 and minor 2 of the wave SH1d review of lane page)."""
    res = {"ref_walk_E8": {"skate": {"ok": None, "exempt": "built on a motion-capture recording", "contacts": 45},
                           "self_contact": {"ok": True, "contacts_exempt": [{"with": "left thigh"}]}},
           "x_g2": {"continuity": {"ok": None, "not_judged": S.SPANS_UNKNOWN.format(key="events_s")},
                    "skate": {"ok": False}}}
    rec = S.checks_record(res)
    nj = rec["skeleton_checks_not_judged"]
    assert nj["skate"]["exempt"]["ref_walk_E8"]["contacts"] == 45 and "ref_walk_E8" in nj["self_contact"]["exempt"]
    assert nj["continuity"] == {"spans_unknown": ["x_g2"]} and rec["skeleton_checks_flagged"]["skate"] == 1
    assert S.DECLARED_SPANS == {"continuity": "events_s", "skate": "slides_s"}
