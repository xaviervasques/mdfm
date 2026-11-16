#!/usr/bin/env python
"""T0.7b of the MD_FM execution plan: medication and group tables for three sources.

(1) weargait_pd: a levodopa-equivalent daily dose (LEDD) per PD subject, parsed from the
    free medication text of the raw subject table ("Current Medications" and "PD
    Medication Dose" of "PD - Demographic+Clinical - datasetV1.csv"), with the conversion
    factors of LEDD_FACTORS below (Tomlinson et al. 2010, extended where needed). Every
    PD drug entry is written with its parse, its rule decisions and its flags, so that a
    reader can audit or recompute any subject. DBS fields and the time since the last
    dose are carried from the same table.
(2) care_pd (evaluation-only): only the raw pickles' subject keys, walk names and the
    per-walk metadata fields (medication, other, fps, number of frames) are read; the
    medication state is parsed from the walk names and compared with the medication field.
    No UPDRS label is read into the outputs.
(3) pd_biostamprc21 (EVAL): group, sex and age per subject from the raw clinic CSV; the
    sensor placement per subject and sensor from the raw zip and the source
    documentation (as quoted by the MoveAtlas ingester); the tagged medication windows from
    the per-subject annotation files; and a measured check of the placement labels: on the
    lateralized UPDRS item windows, the sensor on the tested side should move more than
    the opposite one (left/right check), and the forearm sensors more than the thigh
    sensors during hand items (segment check). The check reads each accelerometer CSV
    inside the zip only up to the last annotated window.

The MoveAtlas tree is read-only. Outputs go to reports/nuisance/ (shared with T0.7a and
T0.7c, so the provenance file is named _provenance_probe_medication.json).

Usage: python scripts/probe_medication.py [--moveatlas PATH] [--out DIR]
"""
import collections
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if "--out" not in sys.argv:
    sys.argv += ["--out", os.path.join(os.path.dirname(HERE), "reports", "nuisance")]
from _common import moveatlas_head, parse_args  # noqa: E402

T_START = time.time()

# ---------------------------------------------------------------------------------------
# LEDD conversion factors
# ---------------------------------------------------------------------------------------
TOMLINSON_2010 = ("Tomlinson CL, Stowe R, Patel S, Rick C, Gray R, Clarke CE. Systematic review of "
                  "levodopa dose equivalency reporting in Parkinson's disease. Mov Disord "
                  "2010;25(15):2649-2653. doi:10.1002/mds.23429")
JOST_2023 = ("Jost ST, Kaldenbach MA, Antonini A, et al. Levodopa dose equivalency in Parkinson's "
             "disease: updated systematic review and proposals. Mov Disord 2023;38(7):1236-1252. "
             "doi:10.1002/mds.29410")
# key: (label, kind, factor, source, status)
#   kind "mg": LEDD = factor x mg/day of the drug
#   kind "levodopa": LEDD = factor x mg/day of levodopa in the product
#   kind "comt": LEDD = factor x levodopa LEDD taken with it (see compute_ledd)
#   kind "none": listed, no dopaminergic LEDD
# status "tomlinson_table": the value of Tomlinson 2010 Table 1 (standard, widely reproduced);
# status "extension_unverified": value used for a product Tomlinson 2010 does not cover, taken
#   from Jost 2023 (or Schade 2020, which Jost 2023 adopts) as recalled; neither paper is on
#   this machine and there is no network, so these values must be checked against the paper
#   before any LEDD enters a claim. The raw mg per class are written beside every LEDD so a
#   corrected factor is a one-line recomputation.
LEDD_FACTORS = {
    "ld_ir": ("levodopa immediate release (incl. orally disintegrating)", "levodopa", 1.0, TOMLINSON_2010, "tomlinson_table"),
    "ld_cr": ("levodopa controlled release (Sinemet CR, Madopar CR)", "levodopa", 0.75, TOMLINSON_2010, "tomlinson_table"),
    "ld_er_ipx066": ("levodopa extended-release capsules (IPX066, Rytary)", "levodopa", 0.5, JOST_2023, "extension_unverified"),
    "lcig": ("levodopa-carbidopa intestinal gel (Duodopa, Duopa)", "levodopa", 1.11, TOMLINSON_2010, "tomlinson_table"),
    "stalevo": ("levodopa-carbidopa-entacapone (Stalevo): levodopa x1 plus its entacapone share", "levodopa", 1.0, TOMLINSON_2010, "tomlinson_table"),
    "entacapone": ("entacapone (also inside Stalevo): levodopa LEDD x 0.33", "comt", 0.33, TOMLINSON_2010, "tomlinson_table"),
    "tolcapone": ("tolcapone: levodopa LEDD x 0.5", "comt", 0.5, TOMLINSON_2010, "tomlinson_table"),
    "opicapone": ("opicapone: levodopa LEDD x 0.5", "comt", 0.5, JOST_2023, "extension_unverified"),
    "pramipexole": ("pramipexole (as salt, IR and ER)", "mg", 100.0, TOMLINSON_2010, "tomlinson_table"),
    "ropinirole": ("ropinirole (IR and prolonged release)", "mg", 20.0, TOMLINSON_2010, "tomlinson_table"),
    "rotigotine": ("rotigotine patch (mg per 24 h)", "mg", 30.0, TOMLINSON_2010, "tomlinson_table"),
    "apomorphine": ("apomorphine", "mg", 10.0, TOMLINSON_2010, "tomlinson_table"),
    "bromocriptine": ("bromocriptine", "mg", 10.0, TOMLINSON_2010, "tomlinson_table"),
    "cabergoline": ("cabergoline", "mg", 80.0, TOMLINSON_2010, "tomlinson_table"),
    "pergolide": ("pergolide", "mg", 100.0, TOMLINSON_2010, "tomlinson_table"),
    "lisuride": ("lisuride", "mg", 100.0, TOMLINSON_2010, "tomlinson_table"),
    "piribedil": ("piribedil", "mg", 1.0, TOMLINSON_2010, "tomlinson_table"),
    "selegiline": ("selegiline oral", "mg", 10.0, TOMLINSON_2010, "tomlinson_table"),
    "selegiline_sl": ("selegiline sublingual", "mg", 80.0, TOMLINSON_2010, "tomlinson_table"),
    "rasagiline": ("rasagiline", "mg", 100.0, TOMLINSON_2010, "tomlinson_table"),
    "safinamide": ("safinamide (100 mg counted as 150 mg LED)", "mg", 1.5, JOST_2023, "extension_unverified"),
    "amantadine": ("amantadine (factor 1 also applied to extended-release amantadine, flagged)", "mg", 1.0, TOMLINSON_2010, "tomlinson_table"),
    "carbidopa": ("carbidopa alone (Lodosyn): no dopaminergic effect", "none", 0.0, "no factor (not dopaminergic)", "not_converted"),
    "trihexyphenidyl": ("trihexyphenidyl (anticholinergic)", "none", 0.0, "not in Tomlinson 2010", "not_converted"),
    "benztropine": ("benztropine (anticholinergic)", "none", 0.0, "not in Tomlinson 2010", "not_converted"),
    "droxidopa": ("droxidopa (noradrenergic prodrug)", "none", 0.0, "not in Tomlinson 2010", "not_converted"),
    "istradefylline": ("istradefylline", "none", 0.0, "not converted in this script", "not_converted"),
}
LEVODOPA_KEYS = ("ld_ir", "ld_cr", "ld_er_ipx066", "lcig", "stalevo")
COMT_KEYS = ("entacapone", "tolcapone", "opicapone")
# drugs whose label frequency is once daily only; a missing frequency is filled with 1/day (info flag)
ONCE_DAILY_BY_LABEL = ("rasagiline", "rotigotine", "safinamide", "opicapone")

# lexicon: first match on the product part of an entry wins (order matters)
DRUG_LEXICON = [
    ("stalevo", r"stalevo|carbidopa-levodopa-entacapone|levodopa-carbidopa-entacapone"),
    ("lcig", r"duopa|duodopa|intestinal gel|^pump\b"),
    ("ld_er_ipx066", r"rytary|rytari|numient|ipx066|levodopa.*extended[- ]release capsule|levodopa.*\ber capsule"),
    ("ld_cr", r"levodopa.*(?:\bcr\b|controlled|\ber tablet|sustained)|sinemet cr|madopar cr|madopar hbs"),
    ("ld_ir", r"levodopa|levadopa|sinemet|siment|cynamet|parcopa|madopar|dhivy"),
    ("carbidopa", r"lodosyn|\bcarbidopa\b"),
    ("entacapone", r"entacapone|comtan|antofecpone"),
    ("tolcapone", r"tolcapone|tasmar"),
    ("opicapone", r"opicapone|ongentys"),
    ("pramipexole", r"pramipexole|mirapex"),
    ("ropinirole", r"ropinirole|requip|rapinirole"),
    ("rotigotine", r"rotigotine|neupro"),
    ("apomorphine", r"apomorphine|kynmobi|apokyn"),
    ("bromocriptine", r"bromocriptine|parlodel"),
    ("cabergoline", r"cabergoline"),
    ("rasagiline", r"rasagiline|rasagaline|azilect"),
    ("selegiline_sl", r"zelapar"),
    ("selegiline", r"selegiline|seligilene|eldepryl"),
    ("safinamide", r"safinamide|xadago"),
    ("amantadine", r"amantadine|symmetrel|gocovri|osmolex"),
    ("trihexyphenidyl", r"trihexyphenidyl|artane"),
    ("benztropine", r"benztropine|cogentin"),
    ("droxidopa", r"droxidopa|northera"),
    ("istradefylline", r"istradefylline|nourianz"),
]
DRUG_RES = [(k, re.compile(p, re.I)) for k, p in DRUG_LEXICON]
# non-PD names met in the free-text (tab-less, verb-less) medication fields; any other unrecognized
# item of such a field may be a PD drug the lexicon does not know, so it makes the LEDD a lower bound
NON_PD_KNOWN_RE = re.compile(
    r"clonazepam|vitamin|\bb12\b|pyridostigmine|flu?o?drocortisone|midodrine|\biron\b|levothyrox|lextheroxin|"
    r"mirabegron|captopril|memantine|omepr|quetiapine|rivas?t?igmine|zo[li]edronic|gabapentin|trazodone|"
    r"mi[rt]+azapine|blood pressure|heart|\bpimp\b|lexapro|lamictal|^dbs$|^pump \+ dbs$", re.I)

# flags that leave a number in place but ask for a human look before it is used
REVIEW_FLAGS = {
    "expired_kept_no_pd_dose_list", "expired_kept_listed_in_pd_dose_field", "conflicting_duplicates",
    "alternative_sigs", "replace_or_substitute_note", "titration_or_then_note", "time_limited_course",
    "q_hours_awake16", "q_hours_24", "co_dose_note", "sig_mentions_unlisted_pd_drug", "range_low_taken",
    "prn_only_entry", "comt_without_levodopa", "amantadine_er_factor_as_ir",
}

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
UNIT = r"(?:tablets?|tabs?|capsules?|caps?|cp24|patch(?:es)?|films?|pills?)"
UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*" + UNIT + r"\b")
RANGE_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*" + UNIT + r"\b")
TOTAL_MG_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*(mg|mcg)\s*(?:total|per dose)\)")
VERB_MG_RE = re.compile(r"^(?:take|place|give|apply|dissolve)\s+(\d+(?:\.\d+)?)\s*(mg|mcg)\b")
TIME_RE = re.compile(r"\b(\d{1,2}(?::\d{2})?|\d{3,4})\s*(am|pm|a|p)\b")
VERB_RE = re.compile(r"^(?:take|place|dissolve|apply|inject|use|give|chew|swallow)\b")
PRN_RE = re.compile(r"as needed|\bprn\b|if needed|may also take|may take an additional|may have an additional|"
                    r"\bup to\b|if excessive|breakthrough")
DAILY_RE = re.compile(r"\b(?:daily|once a day|once daily|every day|nightly|at night|at bedtime|bedtime|every morning|"
                      r"every evening|in the morning|in the evening|each morning|each evening|every night|overnight|"
                      r"around noon|at noon|qhs|qam|qpm|@ hs|at hs|\bhs\b)\b|\bhs\b")
SPLIT_ENTRY_RE = re.compile(r"(?:(?<=\t)|(?<=\.\s)|(?<=\d))(?=[A-Za-z][A-Za-z\-]+(?: [A-Za-z][\w\-]*){0,3} "
                            r"\((?!Expired)[A-Za-z][^()]{0,40}\) ?\d)")
SIG_VERB_SPLIT_RE = re.compile(r"\s(?=(?:Take|TAKE|Place|PLACE|Dissolve|DISSOLVE|Apply|APPLY|Inject|INJECT)\b)")


def normalize(s):
    s = s.replace("\xa0", " ").lower()
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)
    s = re.sub(r"\bone (?:and|&) (?:one-half|a half|1/2)\b", "1.5", s)
    s = re.sub(r"\b(\d+)\s*(?:and|&)\s*1/2\b", lambda m: str(int(m.group(1)) + 0.5), s)
    s = re.sub(r"\b(\d+) 1/2\b", lambda m: str(int(m.group(1)) + 0.5), s)
    s = re.sub(r"\b1/2\b", "0.5", s)
    s = re.sub(r"\bone-half\b", "0.5", s)
    s = re.sub(r"\b(" + "|".join(WORDNUM) + r")\b", lambda m: str(WORDNUM[m.group(1)]), s)
    return s


def freq_of(p):
    """Administrations per day in one normalized part; (freq, flag) or (None, None)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\(\d+\)\s*)?(?:times|x)\s*(?:/\s*)?(?:a|per|each)?\s*week", p)
    if m:
        return float(m.group(1)) / 7.0, "weekly"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\(\d+\)\s*)?(?:times|x)\s*(?:/\s*)?(?:a|per|each)?\s*(?:day|daily)\b", p)
    if m:
        return float(m.group(1)), None
    if re.search(r"\bqid\b", p):
        return 4.0, None
    if re.search(r"\btid\b", p):
        return 3.0, None
    if re.search(r"\bbid\b|\btwice\b", p):
        return 2.0, None
    if re.search(r"\bthrice\b", p):
        return 3.0, None
    m = re.search(r"(?:every|\bq)\s*~?(\d+(?:\.\d+)?)(?:\s*(?:-|to)\s*\d+)?\s*(?:\(\d+\)\s*)?(?:hours?|hrs?|h)\b", p)
    if m:
        n = float(m.group(1))
        if "awake" in p:
            return float(math.floor(16.0 / n) + 1), "q_hours_awake16"
        return 24.0 / n, "q_hours_24"
    times = TIME_RE.findall(p)
    if times:
        return float(len(times)), "schedule_times"
    if re.search(r"morning\s*(?:&|and)\s*evening", p):
        return 2.0, None
    if "every other day" in p:
        return 0.5, None
    if DAILY_RE.search(p):
        return 1.0, None
    return None, None


def qty_of(p):
    """Amount per administration in one normalized part: ('units'|'mg', value, flag) or None."""
    m = RANGE_UNIT_RE.search(p)
    if m:
        return ("units", float(m.group(1)), "range_low_taken")
    m = UNIT_RE.search(p)
    if m:
        return ("units", float(m.group(1)), None)
    m = TOTAL_MG_RE.search(p)
    if m:
        v = float(m.group(1)) / (1000.0 if m.group(2) == "mcg" else 1.0)
        return ("mg", v, None)
    m = VERB_MG_RE.search(p)
    if m:
        v = float(m.group(1)) / (1000.0 if m.group(2) == "mcg" else 1.0)
        return ("mg", v, None)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mg|mcg)\b", p)
    if m:
        return ("mg", float(m.group(1)), None)
    return None


def split_parts(sentence):
    """Split one sentence into dose parts at 'and'/',' boundaries that start a new amount."""
    parts = re.split(r"\s+and\s+(?=\d+(?:\.\d+)?\s*" + UNIT + r")|,\s*(?:and\s+)?(?=\d+(?:\.\d+)?\s*" + UNIT + r")"
                     r"|,?\s+(?:and\s+)?(?=may (?:also |take |have ))",
                     sentence)
    return [x.strip() for x in parts if x and x.strip()]


def parse_regimen(sig_raw):
    """Daily amount from a sig. Returns dict(amount, unit, flags, n_parts, prn_parts)."""
    flags = set()
    out = {"amount": None, "unit": None, "flags": flags}
    if not sig_raw or not sig_raw.strip():
        return out
    sig = normalize(sig_raw)
    # dash lists of doses through the day: "3 caps - 2 cap - 3 cap" or "780 mg - 780 mg - 685 mg"
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg(?:\s*-\s*\d+(?:\.\d+)?\s*mg){2,}", sig)
    if m:
        vals = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mg", m.group(0))]
        out.update(amount=sum(vals), unit="mg")
        flags.add("dose_list")
        return out
    m = re.search(r"(\d+(?:\.\d+)?)\s*" + UNIT + r"(?:\s*-\s*\d+(?:\.\d+)?\s*" + UNIT + r"){1,}", sig)
    if m:
        vals = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*" + UNIT, m.group(0))]
        out.update(amount=sum(vals), unit="units")
        flags.add("dose_list")
        return out
    groups = [g for g in sig.split(";") if g.strip()]
    group_results = []
    for g in groups:
        sentences = [s.strip() for s in re.split(r"(?<![0-9])\.\s*(?=[a-z0-9@(])|(?<=[a-z)])\.$", g) if s.strip()]
        verb_parts, other_parts = [], []
        for s in sentences:
            is_verb = bool(VERB_RE.match(s))
            for p in split_parts(s):
                (verb_parts if is_verb else other_parts).append(p)
        res = None
        for pool, tag in ((verb_parts, "verb"), (other_parts, "non_verb")):
            total, unit, used, pflags = 0.0, None, 0, set()
            for p in pool:
                if PRN_RE.search(p):
                    pflags.add("prn_excluded")
                    continue
                q = qty_of(p)
                f, ff = freq_of(p)
                if q is None or f is None:
                    continue
                if unit is not None and q[0] != unit:
                    pflags.add("mixed_units_part_ignored")
                    continue
                unit = q[0]
                total += q[1] * f
                used += 1
                if q[2]:
                    pflags.add(q[2])
                if ff:
                    pflags.add(ff)
            if used:
                res = (total, unit, pflags, tag)
                break
        if res is None:
            # an amount without frequency, or a frequency without amount
            prn_all = all(PRN_RE.search(p) for p in verb_parts + other_parts) if (verb_parts or other_parts) else False
            q_any = next((qty_of(p) for p in verb_parts + other_parts if qty_of(p)), None)
            f_any = next((freq_of(p) for p in verb_parts + other_parts if freq_of(p)[0] is not None), (None, None))
            group_results.append({"total": None, "prn_all": prn_all, "q": q_any, "f": f_any[0]})
        else:
            group_results.append({"total": res[0], "unit": res[1], "flags": res[2], "pool": res[3]})
    done = [r for r in group_results if r.get("total") is not None]
    if done:
        if len(done) > 1:
            flags.add("alternative_sigs")
        best = max(done, key=lambda r: r["total"])
        out.update(amount=best["total"], unit=best["unit"])
        flags |= best["flags"]
        if best["pool"] == "non_verb":
            flags.add("sig_without_instruction_verb")
        for r in group_results:
            if r.get("flags"):
                flags |= {x for x in r["flags"] if x == "prn_excluded"}
        return out
    # nothing complete
    if group_results and all(r.get("prn_all") for r in group_results):
        flags.add("prn_only_entry")
        out.update(amount=0.0, unit="units")
        return out
    q = next((r["q"] for r in group_results if r.get("q")), None)
    f = next((r["f"] for r in group_results if r.get("f") is not None), None)
    out["partial_q"] = q
    out["partial_f"] = f
    return out


def identify(product):
    for key, rx in DRUG_RES:
        if rx.search(product):
            return key
    return None


def parse_strength(product, key):
    p = product.lower().replace(",", "")
    if key in LEVODOPA_KEYS:
        m = re.search(r"(\d+(?:\.\d+)?)\s*[-/]\s*(\d+(?:\.\d+)?)(?:\s*[-/]\s*(\d+(?:\.\d+)?))?\s*mg", p)
        if m:
            return {"ld": float(m.group(2)), "cd": float(m.group(1)),
                    "ent": float(m.group(3)) if m.group(3) else None}
        m = re.search(r"(\d+(?:\.\d+)?)\s*mg", p)
        if m:
            return {"ld": float(m.group(1)), "cd": None, "ent": None}
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg)", p)
    if m:
        return {"mg": float(m.group(1)) / (1000.0 if m.group(2) == "mcg" else 1.0)}
    return None


def split_product_sig(entry):
    """Epic lines: product before the first tab. Otherwise before the first instruction verb."""
    e = entry.strip()
    if "\t" in e:
        prod, sig = e.split("\t", 1)
        return prod.strip(), sig.replace("\t", " ").strip()
    parts = SIG_VERB_SPLIT_RE.split(e, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return e, e


def split_entries(text, field):
    """Split a medication field into entries. Returns list of (entry_text, is_clinic_administered)."""
    if text is None:
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    clinic = False
    for line in t.split("\n"):
        ls = line.strip()
        if not ls or ls in ("-", "NA", "N/A"):
            continue
        if re.match(r"clinic-administered medications", ls, re.I):
            clinic = True
            continue
        if re.match(r"outpatient medications$", ls, re.I):
            continue
        segs = SPLIT_ENTRY_RE.split(ls)
        for seg in segs:
            seg = seg.strip().strip("\t").strip()
            if not seg:
                continue
            if "\t" not in seg and "\t" not in t:
                # free text: normalize 'carbidopa/levodopa' then split comma and slash lists
                s2 = re.sub(r"car[bv][io]dopa\s*[/ ]\s*lev[ao]dopa", "carbidopa-levodopa", seg, flags=re.I)
                items = re.split(r",\s+(?=[A-Za-z])|(?<![0-9]x)(?<=[A-Za-z)])\s*/\s*(?=[A-Za-z])(?!day\b|week\b)", s2)
                for it in items:
                    if it.strip():
                        out.append((it.strip(), clinic))
            else:
                out.append((seg, clinic))
    return out


def parse_field(text, field):
    """All entries of a field with their drug identification and regimen parse."""
    entries = []
    for raw, clinic in split_entries(text, field):
        product, sig = split_product_sig(raw)
        key = identify(product)
        if key is None and "\t" not in raw:
            key = identify(raw)
        e = {"field": field, "raw": raw, "product": product, "sig": sig, "key": key, "clinic": clinic,
             "flags": set(), "drop": None}
        low = raw.lower()
        e["expired"] = "(expired)" in low
        e["not_taking"] = "patient not taking" in low
        e["clinic_dose"] = clinic or bool(re.search(r":\s*dose:", low))
        m = re.search(r"patient taking differently:\s*(.*)$", sig, re.I | re.S)
        if m:
            e["taking_differently"] = True
            sig_used = m.group(1)
        else:
            e["taking_differently"] = False
            sig_used = re.split(r"patient not taking", sig, flags=re.I)[0]
        e["sig_used"] = sig_used
        if key is None:
            free = "\t" not in (text or "") and not re.search(r"(?i)\b(?:take|place|dissolve|apply|inject|instill)\b", text or "")
            e["free_text_item"] = free
            s0 = raw.strip()
            e["unrecognized"] = bool(free and re.search(r"[A-Za-z]{3,}", s0) and not re.match(r"(?i)^(?:\d|half\b)", s0)
                                     and not NON_PD_KNOWN_RE.search(s0))
        if key is not None:
            e["strength"] = parse_strength(product, key)
            reg = parse_regimen(sig_used)
            e["regimen"] = reg
            e["flags"] |= reg["flags"]
            nl = normalize(sig_used)
            if re.search(r"\breplace|substitut|alternat", nl):
                e["flags"].add("replace_or_substitute_note")
            if re.search(r"\bthen\b|titrat|taper", nl):
                e["flags"].add("titration_or_then_note")
            if re.search(r"\bfor \d+ days\b|\bfor \d+ weeks\b|x \d+ days", nl):
                e["flags"].add("time_limited_course")
            if re.search(r"along with|take with a|with a \d|in addition to|\bwith 1 (?:cap|tab)", nl):
                e["flags"].add("co_dose_note")
            if sig_used.strip() and sig_used.strip() != product.strip():
                e["sig_mentions"] = sorted({k for k, rx in DRUG_RES if k != key and k != "carbidopa" and rx.search(sig_used)})
        entries.append(e)
    return entries


def product_id(e):
    s = e.get("strength") or {}
    if e["key"] in LEVODOPA_KEYS:
        return (e["key"], s.get("ld"))
    return (e["key"], s.get("mg"))


def daily_mg(e):
    """(mg/day of the active drug or of levodopa, flags) or (None, flags)."""
    key, reg, s = e["key"], e.get("regimen") or {}, e.get("strength")
    fl = set()
    amount, unit = reg.get("amount"), reg.get("unit")
    if amount is None and key in ONCE_DAILY_BY_LABEL:
        q = reg.get("partial_q")
        if q is not None and reg.get("partial_f") is None:
            amount, unit = q[1], q[0]
            fl.add("label_frequency_assumed")
        elif q is None and reg.get("partial_f") is None and s is not None:
            amount, unit = 1.0, "units"
            fl.add("label_frequency_assumed")
            fl.add("one_unit_implied")
    if amount is None:
        # free-text items: strength plus frequency, no count ("Amantadine 100 mg TID")
        q, f = reg.get("partial_q"), reg.get("partial_f")
        if f is not None and s is not None and (q is None or (q[0] == "mg")):
            amount, unit = f, "units"
            fl.add("one_unit_implied")
    if amount is None:
        return None, fl
    if unit == "mg":
        return amount, fl
    if s is None:
        return None, fl | {"strength_missing"}
    if key in LEVODOPA_KEYS:
        return amount * s["ld"], fl
    return amount * s["mg"], fl


def weargait_subject(sid, current, pddose):
    cur = parse_field(current, "current_medications")
    pdd = parse_field(pddose, "pd_medication_dose")
    pd_cur = [e for e in cur if e["key"]]
    pd_pdd = [e for e in pdd if e["key"]]
    subj_flags = set()
    unrecognized = []
    pdd_products = {product_id(e) for e in pd_pdd if not e["not_taking"]}
    pdd_has_pd = bool(pd_pdd)
    # choose the primary list
    if pd_cur:
        primary, primary_field = pd_cur, "current_medications"
        cur_keys_parsed = set()
        for e in pd_cur:
            mg, _ = daily_mg(e)
            if mg is not None:
                cur_keys_parsed.add(e["key"])
        cur_keys = {e["key"] for e in pd_cur}
        extra = []
        for e in pd_pdd:
            if e["clinic_dose"] or e["not_taking"]:
                continue
            mg, _ = daily_mg(e)
            if e["key"] not in cur_keys:
                e["flags"].add("from_pd_dose_field")
                extra.append(e)
            elif e["key"] not in cur_keys_parsed and mg is not None:
                e["flags"].add("from_pd_dose_field_replaces_undosed_current")
                for c in pd_cur:
                    if c["key"] == e["key"]:
                        c["drop"] = "superseded_by_dosed_pd_dose_field_entry"
                extra.append(e)
        entries = pd_cur + extra
        used_fields = [cur] + ([pdd] if extra else [])
    elif pd_pdd:
        entries, primary_field = pd_pdd, "pd_medication_dose"
        used_fields = [pdd]
    else:
        entries, primary_field = [], ""
        used_fields = []
    # unrecognized items of a free-text field in use: possibly a PD drug the lexicon does not know
    for fld in used_fields:
        for e in fld:
            if e["key"] is None and e.get("unrecognized"):
                unrecognized.append(e["raw"].strip())
    if unrecognized:
        subj_flags.add("unrecognized_item_in_free_text")
    # drop rules
    for e in entries:
        if e["drop"]:
            continue
        if e["clinic_dose"]:
            e["drop"] = "clinic_administered_dose"
        elif e["not_taking"]:
            e["drop"] = "patient_not_taking"
        elif e["expired"] and not e["taking_differently"]:
            if e["field"] == "current_medications" and pdd_has_pd:
                if product_id(e) in pdd_products:
                    e["flags"].add("expired_kept_listed_in_pd_dose_field")
                else:
                    e["drop"] = "expired_not_in_pd_dose_field"
            elif "time_limited_course" in e["flags"] or "titration_or_then_note" in e["flags"]:
                e["drop"] = "expired_time_limited_course"
            else:
                e["flags"].add("expired_kept_no_pd_dose_list")
    # explicit replacement notes: "REPLACES RYTARY 195 MG", "Replaces Sinemet IR"
    for e in entries:
        if e["drop"] or not e["key"]:
            continue
        for m in re.finditer(r"replaces?\s+([a-z ]+?)\s*(\d+(?:\.\d+)?)?\s*(?:mg)?(?:\s|$|\.|,)", normalize(e["sig_used"])):
            target_key = identify(m.group(1))
            if target_key == "ld_ir" and re.search(r"\bir\b|immediate", m.group(1)) is None and "sinemet" in m.group(1):
                target_key = "ld_ir"
            if target_key is None:
                continue
            tstr = float(m.group(2)) if m.group(2) else None
            for o in entries:
                if o is e or o["drop"] or o["key"] != target_key:
                    continue
                if tstr is not None and product_id(o)[1] != tstr:
                    continue
                o["drop"] = "replaced_by_other_entry"
                e["flags"].add("replace_or_substitute_note")
    kept = [e for e in entries if not e["drop"]]
    # amounts
    for e in kept:
        mg, fl = daily_mg(e)
        e["mg_day"] = mg
        e["flags"] |= fl
        if e["key"] == "amantadine" and re.search(r"gocovri|osmolex|cp24|\ber\b|24 ?hr", e["product"], re.I):
            e["flags"].add("amantadine_er_factor_as_ir")
    # duplicates of the same product
    by_prod = collections.defaultdict(list)
    for e in kept:
        by_prod[product_id(e)].append(e)
    for pid, grp in by_prod.items():
        if len(grp) < 2:
            continue
        vals = [g["mg_day"] for g in grp]
        known = [v for v in vals if v is not None]
        if known and len(set(round(v, 6) for v in known)) == 1:
            keep = min((g for g in grp if g["mg_day"] is not None),
                       key=lambda g: (len(g["flags"] & REVIEW_FLAGS), int(g["expired"]), grp.index(g)))
            for g in grp:
                if g is not keep:
                    g["drop"] = "duplicate_collapsed"
            keep["flags"].add("duplicate_collapsed")
        elif known:
            keep = max((g for g in grp if g["mg_day"] is not None), key=lambda g: g["mg_day"])
            for g in grp:
                if g is not keep:
                    g["drop"] = "conflicting_duplicate_lower_dropped"
            keep["flags"].add("conflicting_duplicates")
    kept = [e for e in entries if not e["drop"]]
    # a sig that names a PD drug no entry of the subject lists ("with Mirapex ER")
    named = {e["key"] for e in entries if e["key"]}
    for e in kept:
        miss = [k for k in e.get("sig_mentions", []) if k not in named and LEDD_FACTORS[k][1] != "none"]
        if miss:
            e["flags"].add("sig_mentions_unlisted_pd_drug")
    return entries, kept, primary_field, subj_flags, unrecognized


def compute_ledd(kept):
    raw = collections.defaultdict(float)
    comp = {"levodopa": 0.0, "comt": 0.0, "agonist": 0.0, "maob": 0.0, "amantadine": 0.0}
    unparsed, counted, comt = [], [], []
    for e in kept:
        k = e["key"]
        kind = LEDD_FACTORS[k][1]
        if kind == "none":
            continue
        if kind == "comt":
            comt.append(e)
            continue
        if e.get("mg_day") is None:
            unparsed.append(e)
            continue
        raw[k] += e["mg_day"]
        counted.append(e)
    ld_non_stalevo = sum(raw[k] * LEDD_FACTORS[k][2] for k in ("ld_ir", "ld_cr", "ld_er_ipx066", "lcig"))
    comp["levodopa"] = ld_non_stalevo + raw["stalevo"]
    comt_factor = max([LEDD_FACTORS[e["key"]][2] for e in comt], default=0.0)
    comp["comt"] = comt_factor * ld_non_stalevo + LEDD_FACTORS["entacapone"][2] * raw["stalevo"]
    for k in ("pramipexole", "ropinirole", "rotigotine", "apomorphine", "bromocriptine", "cabergoline",
              "pergolide", "lisuride", "piribedil"):
        comp["agonist"] += raw[k] * LEDD_FACTORS[k][2]
    for k in ("rasagiline", "selegiline", "selegiline_sl", "safinamide"):
        comp["maob"] += raw[k] * LEDD_FACTORS[k][2]
    comp["amantadine"] = raw["amantadine"] * LEDD_FACTORS["amantadine"][2]
    for e in comt:
        if ld_non_stalevo + raw["stalevo"] == 0:
            e["flags"].add("comt_without_levodopa")
        if not (e.get("regimen") or {}).get("amount"):
            e["flags"].add("comt_regimen_not_parsed")
    return comp, raw, unparsed, counted, comt


def parse_clock(s):
    m = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp])\.?\s*[Mm]?", s or "")
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if h == 12:
        h = 0
    if ap == "p":
        h += 12
    return h + mi / 60.0


def run_weargait(a, out):
    root = f"{a.moveatlas}/datasets/imu/weargait_pd/raw/Version 1"
    pd_path = f"{root}/PD - Demographic+Clinical - datasetV1.csv"
    hc_path = f"{root}/CONTROLS - Demographic+Clinical - datasetV1.csv"
    with open(pd_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    hdr = rows[1]
    col = {h: i for i, h in enumerate(hdr)}
    need = ["Subject ID", "Release version", "Time of research session", "Time of last medication dose",
            "Current Medications", "PD Medication Dose", "DBS?", "Bilateral vs uilateral",
            "Electrode location(s)", "Years since surgery", "Years since PD diagnosis"]
    for n in need:
        if n not in col:
            sys.exit(f"weargait_pd: column {n!r} not found in the PD table header")
    subj_rows, entry_rows = [], []
    for r in rows[2:]:
        if not r or not r[col["Subject ID"]].strip():
            continue
        sid = r[col["Subject ID"]].strip()
        current = r[col["Current Medications"]]
        pddose = r[col["PD Medication Dose"]]
        has_cur = current.strip() not in ("", "-")
        has_pdd = pddose.strip() not in ("", "-")
        entries, kept, primary_field, subj_flags, unrecognized = weargait_subject(sid, current, pddose)
        comp, raw, unparsed, counted, comt = compute_ledd(kept)
        ledd_total = sum(comp.values())
        pd_named = [e for e in entries if e["key"]]
        relevant = [e for e in kept if LEDD_FACTORS[e["key"]][1] != "none"]
        # subject flags: those of the kept entries that carry an LEDD, plus the subject-level ones
        flags = set(subj_flags)
        for e in relevant:
            flags |= e["flags"]
        # two or more strengths of one drug summed (read literally from the list); an entry that adds
        # nothing (a PRN-only entry counted as 0 mg/day) is not a summed strength
        strengths = collections.defaultdict(set)
        for e in counted:
            if e.get("mg_day"):
                strengths[e["key"]].add(product_id(e)[1])
        multi_keys = sorted(k for k, v in strengths.items() if len(v) > 1)
        if any(k in LEVODOPA_KEYS for k in multi_keys):
            flags.add("levodopa_multiple_strengths_summed")
        if any(k not in LEVODOPA_KEYS for k in multi_keys):
            flags.add("other_drug_multiple_strengths_summed")
        drops = collections.Counter(e["drop"] for e in entries if e["drop"])
        # status
        comt_unusable = [e for e in comt if "comt_without_levodopa" in e["flags"]]
        if not has_cur and not has_pdd:
            status = "missing"
        elif not pd_named:
            status = "no_pd_drug_listed"
        elif not relevant:
            status = "no_ledd_drug_current"
        elif not counted:
            status = "illegible"
        elif unparsed or comt_unusable or unrecognized:
            status = "partial"
        elif flags & REVIEW_FLAGS:
            status = "legible_review"
        else:
            status = "legible"
        legible = status in ("legible", "legible_review")
        factor_flags = sorted({e["key"] for e in counted + comt if LEDD_FACTORS[e["key"]][4] == "extension_unverified"})
        dbs = r[col["DBS?"]].strip()
        t_sess = parse_clock(r[col["Time of research session"]])
        t_dose = parse_clock(r[col["Time of last medication dose"]])
        dt_h, dt_flag = "", ""
        if t_sess is not None and t_dose is not None:
            d = t_sess - t_dose
            dt_h = round(d, 2)
            if d < 0:
                dt_flag = "dose_time_after_session_time"
            elif d > 12:
                dt_flag = "interval_over_12h"
        else:
            dt_flag = "time_not_parsed"
        subj_rows.append({
            "subject_id": sid,
            "release": r[col["Release version"]].strip(),
            "id_prefix": re.match(r"[A-Z]+", sid).group(0) if re.match(r"[A-Z]+", sid) else "",
            "current_medications_present": int(has_cur),
            "pd_medication_dose_present": int(has_pdd),
            "primary_field": primary_field,
            "status": status,
            "ledd_mg": round(ledd_total, 1) if legible else "",
            "ledd_lower_bound_mg": round(ledd_total, 1) if status == "partial" else "",
            "ledd_levodopa_mg": round(comp["levodopa"], 1) if counted else "",
            "ledd_comt_mg": round(comp["comt"], 1) if counted else "",
            "ledd_agonist_mg": round(comp["agonist"], 1) if counted else "",
            "ledd_maob_mg": round(comp["maob"], 1) if counted else "",
            "ledd_amantadine_mg": round(comp["amantadine"], 1) if counted else "",
            "levodopa_ir_mg_raw": round(raw["ld_ir"], 1) if counted else "",
            "levodopa_cr_mg_raw": round(raw["ld_cr"], 1) if counted else "",
            "levodopa_er_ipx066_mg_raw": round(raw["ld_er_ipx066"], 1) if counted else "",
            "levodopa_stalevo_mg_raw": round(raw["stalevo"], 1) if counted else "",
            "comt_inhibitors": ";".join(sorted({e["key"] for e in comt})),
            "drugs_counted": ";".join(sorted({e["key"] for e in counted})),
            "drugs_named_not_dosed": ";".join(sorted({e["key"] for e in unparsed})),
            "drugs_named_not_converted": ";".join(sorted({e["key"] for e in kept if LEDD_FACTORS[e["key"]][1] == "none"})),
            "drugs_multiple_strengths_summed": ";".join(multi_keys),
            "unrecognized_free_text_items": ";".join(unrecognized),
            "n_pd_entries_named": len(pd_named),
            "n_pd_entries_counted": len(counted) + len(comt),
            "entries_dropped": ";".join(f"{k}:{v}" for k, v in sorted(drops.items())),
            "review_flags": ";".join(sorted(flags & REVIEW_FLAGS)),
            "info_flags": ";".join(sorted(flags - REVIEW_FLAGS)),
            "factors_unverified": ";".join(factor_flags),
            "dbs": dbs,
            "dbs_laterality": r[col["Bilateral vs uilateral"]].strip(),
            "dbs_target": r[col["Electrode location(s)"]].strip().upper() if r[col["Electrode location(s)"]].strip() not in ("-", "") else r[col["Electrode location(s)"]].strip(),
            "dbs_years_since_surgery": r[col["Years since surgery"]].strip(),
            "pd_dose_field_mentions_dbs": int(bool(re.search(r"\bdbs\b", pddose, re.I))),
            "pd_dose_field_mentions_pump": int(bool(re.search(r"\bpump\b|duopa", pddose + " " + current, re.I))),
            "years_since_diagnosis": r[col["Years since PD diagnosis"]].strip(),
            "time_of_last_dose": r[col["Time of last medication dose"]].strip(),
            "time_of_session": r[col["Time of research session"]].strip(),
            "hours_since_last_dose_same_day": dt_h,
            "dose_time_flag": dt_flag,
        })
        for e in entries:
            if not e["key"]:
                continue
            s = e.get("strength") or {}
            reg = e.get("regimen") or {}
            entry_rows.append({
                "subject_id": sid, "field": e["field"], "drug": e["key"],
                "ledd_kind": LEDD_FACTORS[e["key"]][1], "factor": LEDD_FACTORS[e["key"]][2],
                "strength_levodopa_mg": s.get("ld", ""), "strength_mg": s.get("mg", ""),
                "expired": int(e["expired"]), "patient_not_taking": int(e["not_taking"]),
                "taking_differently": int(e["taking_differently"]), "clinic_dose": int(e["clinic_dose"]),
                "daily_amount": "" if reg.get("amount") is None else round(reg["amount"], 3),
                "daily_amount_unit": reg.get("unit") or "",
                "mg_per_day": "" if e.get("mg_day") is None else round(e["mg_day"], 3),
                "dropped": e["drop"] or "",
                "flags": ";".join(sorted(e["flags"])),
                "entry_text": re.sub(r"\s+", " ", e["raw"])[:300],
            })
    # controls: any PD drug named in their medication column (sanity check)
    with open(hc_path, encoding="utf-8", newline="") as f:
        hrows = list(csv.reader(f))
    hh = hrows[1]
    if "Current Medications [type][dosage]" not in hh:
        sys.exit("weargait_pd: column 'Current Medications [type][dosage]' not found in the CONTROLS table header")
    hcol = hh.index("Current Medications [type][dosage]")
    hc_pd = []
    n_hc = 0
    for r in hrows[2:]:
        if not r or not r[0].strip():
            continue
        n_hc += 1
        txt = r[hcol] if hcol is not None else ""
        found = sorted({k for k, rx in DRUG_RES if k not in ("carbidopa",) and rx.search(txt)})
        if found:
            hc_pd.append((r[0].strip(), found))
    write_csv(os.path.join(out, "ledd_weargait_pd.csv"), subj_rows)
    write_csv(os.path.join(out, "ledd_weargait_pd_entries.csv"), entry_rows)
    fac_rows = [{"key": k, "label": v[0], "kind": v[1], "factor": v[2], "status": v[4], "source": v[3]}
                for k, v in LEDD_FACTORS.items()]
    write_csv(os.path.join(out, "ledd_factors.csv"), fac_rows)
    # summary numbers
    df = pd.DataFrame(subj_rows)
    st = df["status"].value_counts().to_dict()
    legible = df[df["status"].isin(["legible", "legible_review"])]
    ledd_vals = pd.to_numeric(legible["ledd_mg"], errors="coerce")
    no_dbs = df[df["dbs"] == "No"]
    no_dbs_leg = no_dbs[no_dbs["status"].isin(["legible", "legible_review"])]
    summ = {
        "n_pd_subjects": int(len(df)),
        "n_by_id_prefix": df["id_prefix"].value_counts().to_dict(),
        "status_counts": {k: int(v) for k, v in st.items()},
        "status_counts_by_prefix": {p: {k: int(v) for k, v in g["status"].value_counts().items()}
                                    for p, g in df.groupby("id_prefix")},
        "n_legible_total": int(len(legible)),
        "ledd_legible_median_mg": round(float(ledd_vals.median()), 1) if len(ledd_vals) else None,
        "ledd_legible_iqr_mg": [round(float(ledd_vals.quantile(0.25)), 1), round(float(ledd_vals.quantile(0.75)), 1)] if len(ledd_vals) else None,
        "ledd_legible_min_max_mg": [round(float(ledd_vals.min()), 1), round(float(ledd_vals.max()), 1)] if len(ledd_vals) else None,
        "n_legible_using_unverified_factor": int((legible["factors_unverified"] != "").sum()),
        "unverified_factor_use_counts": dict(collections.Counter(x for v in legible["factors_unverified"] for x in v.split(";") if x)),
        "dbs_counts": df["dbs"].value_counts().to_dict(),
        "dbs_target_counts": df[df["dbs"] == "Yes"]["dbs_target"].value_counts().to_dict(),
        "dbs_laterality_counts": df[df["dbs"] == "Yes"]["dbs_laterality"].str.strip().value_counts().to_dict(),
        "n_no_dbs": int(len(no_dbs)),
        "n_no_dbs_legible": int(len(no_dbs_leg)),
        "n_no_dbs_status": {k: int(v) for k, v in no_dbs["status"].value_counts().items()},
        "subjects_by_status_not_legible": {s: sorted(g["subject_id"].tolist()) for s, g in df.groupby("status")
                                           if s not in ("legible",)},
        "subjects_with_unrecognized_free_text_items": {r["subject_id"]: r["unrecognized_free_text_items"]
                                                       for r in subj_rows if r["unrecognized_free_text_items"]},
        "n_pd_entries_parsed": int(len(entry_rows)),
        "entry_drop_reasons": dict(collections.Counter(r["dropped"] for r in entry_rows if r["dropped"])),
        "review_flag_subject_counts": dict(collections.Counter(x for v in df["review_flags"] for x in v.split(";") if x)),
        "dose_time_flags": df["dose_time_flag"].value_counts().to_dict(),
        "hours_since_last_dose_median": round(float(pd.to_numeric(df["hours_since_last_dose_same_day"], errors="coerce")[df["dose_time_flag"] == ""].median()), 2),
        "n_hours_since_last_dose_clean": int((df["dose_time_flag"] == "").sum()),
        "n_multiple_strengths_summed": int(df["info_flags"].str.contains("levodopa_multiple_strengths_summed").sum()),
        "subjects_multiple_strengths_summed": sorted(df[df["info_flags"].str.contains("levodopa_multiple_strengths_summed")]["subject_id"].tolist()),
        "subjects_other_drug_multiple_strengths_summed": {
            r["subject_id"]: ";".join(k for k in r["drugs_multiple_strengths_summed"].split(";") if k not in LEVODOPA_KEYS)
            for r in subj_rows if "other_drug_multiple_strengths_summed" in r["info_flags"]},
        "max_raw_levodopa_one_formulation": max(
            ({"subject_id": r["subject_id"], "drug": k, "mg_day": r[c]}
             for r in subj_rows for k, c in (("ld_ir", "levodopa_ir_mg_raw"), ("ld_cr", "levodopa_cr_mg_raw"),
                                              ("ld_er_ipx066", "levodopa_er_ipx066_mg_raw"), ("stalevo", "levodopa_stalevo_mg_raw"))
             if r[c] != ""), key=lambda x: x["mg_day"]),
        "n_controls": n_hc,
        "controls_with_pd_drug_named": [{"subject_id": s, "drugs": d} for s, d in hc_pd],
        "inputs": [pd_path, hc_path],
    }
    print(f"weargait_pd: {len(df)} PD subjects; status {summ['status_counts']}")
    return summ


# ---------------------------------------------------------------------------------------
# care_pd
# ---------------------------------------------------------------------------------------
class SafeUnpickler(pickle.Unpickler):
    """Only numpy array reconstruction and plain containers; anything else is refused."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused class {module}.{name}")


CARE_COHORTS = ["3DGait", "BMCLab", "DNE", "E-LC", "KUL-DT-T", "PD-GaM", "T-LTC", "T-SDU", "T-SDU-PD"]


def med_from_walk_name(cohort, walk):
    """(state, token, subject_number_in_name) parsed from a raw walk name."""
    if cohort == "BMCLab":
        m = re.search(r"(SUB\d+)_(on|off)_walk_\d+", walk)
        if m:
            return m.group(2).upper(), m.group(2), m.group(1)
        return "", "", ""
    if cohort == "E-LC":
        m = re.search(r"FOG-(\d+)-.*?(offmed|postmed[0-9a-z]*|onmed)", walk, re.I)
        if m:
            tok = m.group(2).lower()
            return ("OFF" if tok.startswith("off") else "ON"), tok, m.group(1)
        return "", "", ""
    m = re.search(r"(?:^|[_-])(on|off|offmed|onmed|postmed\w*|premed\w*)(?:[_-]|$)", walk, re.I)
    if m:
        tok = m.group(1).lower()
        return ("OFF" if tok.startswith(("off", "pre")) else "ON"), tok, ""
    return "", "", ""


def run_care_pd(a, out):
    root = f"{a.moveatlas}/datasets/pose/care_pd"
    walk_rows, subj_rows = [], []
    for cohort in CARE_COHORTS:
        path = f"{root}/raw/{cohort}.pkl"
        t0 = time.time()
        with open(path, "rb") as f:
            d = SafeUnpickler(f).load()
        for skey in sorted(d, key=lambda x: (len(str(x)), str(x))):
            walks = d[skey]
            states = collections.Counter()
            groups = set()
            name_id_mismatch = 0
            for wid in sorted(walks):
                rec = walks[wid]
                st, tok, nid = med_from_walk_name(cohort, wid)
                std = {"BMCLab": r"SUB\d+_(?:on|off)_walk_\d+", "E-LC": r"FOG-\d+-(?:offmed|postmed[0-9a-z]*)-.*"}.get(cohort)
                variant = int(bool(std) and bool(st) and not re.fullmatch(std, wid))
                field = rec.get("medication")
                field_u = str(field).upper() if field is not None else ""
                agree = "" if (not st and not field_u) else int(st == field_u)
                if nid:
                    if cohort == "BMCLab":
                        ok = nid == str(skey)
                    else:
                        ok = int(nid) == int(skey) if str(skey).isdigit() else False
                    if not ok:
                        name_id_mismatch += 1
                grp = rec.get("other")
                groups.add("" if grp is None else str(grp))
                pose = rec.get("pose")
                nfr = int(pose.shape[0]) if hasattr(pose, "shape") else ""
                walk_rows.append({"cohort": cohort, "subject_key": str(skey), "walk_id": wid,
                                  "med_state_from_name": st, "name_token": tok, "name_variant": variant,
                                  "medication_field": "" if field is None else str(field),
                                  "name_agrees_with_field": agree, "group_field": "" if grp is None else str(grp),
                                  "fps": rec.get("fps", ""), "n_frames": nfr})
                states[st or "none"] += 1
            subj_rows.append({"cohort": cohort, "subject_key": str(skey), "group_field": ";".join(sorted(groups)),
                              "n_walks": sum(states.values()), "n_off": states["OFF"], "n_on": states["ON"],
                              "n_no_state": states["none"], "has_off_and_on": int(states["OFF"] > 0 and states["ON"] > 0),
                              "walk_name_id_mismatches": name_id_mismatch})
        print(f"care_pd {cohort}: {len(d)} subjects, {time.time() - t0:.1f} s", flush=True)
        del d
    write_csv(os.path.join(out, "care_pd_med_state.csv"), walk_rows)
    write_csv(os.path.join(out, "care_pd_med_state_subjects.csv"), subj_rows)
    # cross-check with the MoveAtlas processed manifest (walk set)
    man = pd.read_csv(f"{root}/processed/manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8")
    man_set = set(zip(man["cohort"], man["subject"], man["walk"]))
    raw_set = {(r["cohort"], r["subject_key"], r["walk_id"]) for r in walk_rows}
    wdf, sdf = pd.DataFrame(walk_rows), pd.DataFrame(subj_rows)
    per_cohort = {}
    for c in CARE_COHORTS:
        w, s = wdf[wdf.cohort == c], sdf[sdf.cohort == c]
        pc = {"n_subjects": int(len(s)), "n_walks": int(len(w)),
              "walks_OFF": int((w.med_state_from_name == "OFF").sum()), "walks_ON": int((w.med_state_from_name == "ON").sum()),
              "walks_no_state": int((w.med_state_from_name == "").sum()),
              "subjects_with_OFF": int((s.n_off > 0).sum()), "subjects_with_ON": int((s.n_on > 0).sum()),
              "subjects_with_OFF_and_ON": int(s.has_off_and_on.sum()),
              "walks_medication_field_set": int((w.medication_field != "").sum()),
              "walks_name_field_disagree": int((w.name_agrees_with_field == 0).sum()),
              "walk_name_id_mismatches": int(s.walk_name_id_mismatches.sum()),
              "walks_nonstandard_name_parsed": int(w.name_variant.sum()),
              "name_tokens": w[w.name_token != ""].name_token.value_counts().to_dict()}
        if (w.group_field != "").any():
            pc["subjects_with_OFF_and_ON_by_group"] = {g: int(v) for g, v in s[s.has_off_and_on == 1].group_field.value_counts().items()}
            pc["subjects_by_group"] = {g: int(v) for g, v in s.group_field.value_counts().items()}
        per_cohort[c] = pc
    summ = {"n_walks": len(walk_rows), "n_subjects": len(subj_rows), "per_cohort": per_cohort,
            "manifest_walks": len(man_set), "raw_walks_not_in_manifest": len(raw_set - man_set),
            "manifest_walks_not_in_raw": len(man_set - raw_set),
            "inputs": [f"{root}/raw/{c}.pkl" for c in CARE_COHORTS] + [f"{root}/processed/manifest.csv"]}
    print(f"care_pd: {len(walk_rows)} walks, {len(subj_rows)} subjects")
    return summ


# ---------------------------------------------------------------------------------------
# pd_biostamprc21
# ---------------------------------------------------------------------------------------
# Placement as documented by the source and quoted verbatim by the MoveAtlas ingester
# (ingest_pd_biostamprc21.py, docstring and SENSORS table: "from the dataset page and Adams et
# al. 2021 (npj Parkinson's Disease, 'A real-world study of wearable sensors in Parkinson's
# disease')"). The raw zip itself carries no README; its file names carry only the codes.
BIOSTAMP_SENSORS = {
    "ch": ("chest", "trunk", 0),
    "ll": ("left anterior thigh", "thigh_L", 0),
    "rl": ("right anterior thigh", "thigh_R", 0),
    "lh": ("left anterior forearm", "wrist_L", 1),
    "rh": ("right anterior forearm", "wrist_R", 1),
}
PLACEMENT_SOURCE = ("MoveAtlas ingest_pd_biostamprc21.py docstring and SENSORS table, citing the IEEE DataPort "
                    "page (doi:10.21227/g2g8-1503) and Adams et al. 2021 npj Parkinson's Disease")
ATTR_TYPES = {"area of the body", "arm", "direction", "foot", "hand", "leg", "medication", "side",
              "what side is the participant lying on?"}
HAND_TASKS = {"UPDRS 3.4 - Finger Tapping", "UPDRS 3.5 - Hand Movements", "UPDRS 3.6 - Pronation/Supination",
              "UPDRS 3.16 - Kinetic Tremor of Hands", "UHDRS 6 - Finger Taps", "UHDRS 7 - Pronate/Supinate"}
LEG_TASKS = {"UPDRS 3.7 - Toe Tapping", "UPDRS 3.8 - Leg Agility"}
# alignment check of the annotation clock against the sensor clock: the chest and thighs should move
# far more in the walking windows than in the seated, still windows
WALK_TASKS = {"UPDRS 3.10 - Gait", "Timed Up and Go", "10 Meter Walk Test", "UHDRS 13 - Gait",
              "UHDRS 14 - Tandem Walking", "Smartphone - Gait Test"}
STILL_TASKS = {"UPDRS 3.17 - Rest Tremor Amplitude", "UPDRS 3.1 - Speech", "UPDRS 3.2 - Facial Expression",
               "Smartphone - Voice Test", "UHDRS 1 - Ocular Pursuit", "UHDRS 2 - Saccade Initiation",
               "UHDRS 3 - Saccade Velocity", "UHDRS 4 - Dysarthria", "UHDRS 5 - Tongue Protrusion"}
ALIGN_MIN_LOG10 = 1.0
LAT_MIN_WINDOWS = 4
LAT_CONSISTENT = 0.75


def parse_annotations(rows):
    """Task windows with their attribute rows (Medication, Hand, Side, Foot, Leg, Area of the body, ...).

    In these files an attribute row has start == stop == the stop timestamp of the task row it describes,
    but the rows are not always in sequence: an attribute row can sit before its task row or after the
    next task row (for example the Foot row of a Toe Tapping window listed after the following Leg Agility
    row). Each attribute row is therefore attached to the task window whose stop timestamp equals its own
    timestamp, not to the preceding task row. Identical task rows (same type, start and stop) are one
    window. An attribute row whose timestamp matches no task stop falls back to the preceding task row
    (counted), or is an orphan when no task row precedes it. Two different values of one attribute on
    one window are counted and the value is set to CONFLICT (never read as a side or a state).
    Returns (windows, stats)."""
    windows, by_key, by_stop = [], {}, collections.defaultdict(list)
    stats = collections.Counter()
    parsed = []
    for r in rows:
        if len(r) < 4:
            stats["short_rows"] += 1
            continue
        et = r[0].strip()
        key = et.rstrip(":").strip().lower()
        a0, b0 = int(float(r[1])), int(float(r[2]))
        if key in ATTR_TYPES:
            parsed.append(("attr", key, a0, r[3].strip().upper()))
            continue
        stats["task_rows"] += 1
        wkey = (et, a0, b0)
        if wkey in by_key:
            by_key[wkey]["duplicate_rows"] += 1
            stats["duplicate_task_rows"] += 1
            parsed.append(("task", by_key[wkey]))
            continue
        w = {"task": et, "start_ms": a0, "stop_ms": b0, "attrs": {}, "medication": None, "duplicate_rows": 0,
             "_vals": collections.defaultdict(set)}
        by_key[wkey] = w
        by_stop[b0].append(w)
        windows.append(w)
        parsed.append(("task", w))
    prev = None
    for item in parsed:
        if item[0] == "task":
            prev = item[1]
            continue
        _, key, ts, val = item
        stats["attribute_rows"] += 1
        owners = by_stop.get(ts, [])
        if len(owners) == 1:
            owner = owners[0]
        elif len(owners) > 1:
            stats["attribute_rows_ambiguous_stop"] += 1
            owner = prev if prev in owners else owners[0]
        elif prev is not None:
            stats["attribute_rows_unmatched_fallback"] += 1
            owner = prev
        else:
            stats["orphan_attribute_rows"] += 1
            continue
        if owner is not prev:
            stats["attribute_rows_out_of_sequence"] += 1
        owner["_vals"][key].add(val)
    for w in windows:
        for key, vals in w.pop("_vals").items():
            if len(vals) > 1:
                stats["attribute_conflicts"] += 1
                v = "CONFLICT"
            else:
                v = next(iter(vals))
            if key == "medication":
                w["medication"] = v
            else:
                w["attrs"][key] = v
    return windows, stats


def window_side(w):
    for k in ("hand", "side", "arm", "foot", "leg"):
        v = w["attrs"].get(k)
        if v in ("LEFT", "RIGHT"):
            return v
    return ""


def read_sensor_head(zf, name, max_ms):
    """Timestamps and xyz from the start of one accelerometer CSV up to max_ms (streamed)."""
    nrows = int(max_ms / 32.0) + 20000
    with zf.open(name) as f:
        df = pd.read_csv(f, nrows=nrows, header=0, names=["t", "x", "y", "z"], dtype=np.float64, engine="c")
    return df["t"].to_numpy(), df[["x", "y", "z"]].to_numpy()


def dyn_energy(t, xyz, a, b):
    i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
    if i1 - i0 < 16:
        return None
    seg = xyz[i0:i1]
    seg = seg[np.isfinite(seg).all(axis=1)]
    if len(seg) < 16:
        return None
    return float(seg.var(axis=0).sum())


OFFSET_MIN_S, OFFSET_MAX_S, OFFSET_STEP_S, OFFSET_EXCLUDE_S = -3600, 7200, 5, 60


def measure_placement(sid, status, windows, data, offset_ms):
    """Left/right check per sensor pair, segment check per sensor and clock alignment, with the
    annotation windows shifted by offset_ms onto the sensor clock."""
    lat = {"forearms": collections.Counter(), "thighs": collections.Counter()}
    lat_logr = {"forearms": [], "thighs": []}
    seg = {code: collections.Counter() for code in BIOSTAMP_SENSORS}
    rows = []
    for w in windows:
        side = window_side(w)
        if not side:
            continue
        if w["task"] in HAND_TASKS:
            pname, pair, other_seg = "forearms", ("lh", "rh"), ("ll", "rl")
        elif w["task"] in LEG_TASKS:
            pname, pair, other_seg = "thighs", ("ll", "rl"), ("lh", "rh")
        else:
            continue
        a0, b0 = w["start_ms"] + offset_ms, w["stop_ms"] + offset_ms
        ipsi = pair[0] if side == "LEFT" else pair[1]
        contra = pair[1] if side == "LEFT" else pair[0]
        ipsi_other = other_seg[0] if side == "LEFT" else other_seg[1]
        e = {c: (dyn_energy(*data[c], a0, b0) if c in data else None) for c in ("lh", "rh", "ll", "rl")}
        ei, ec, eo = e[ipsi], e[contra], e[ipsi_other]
        if ei is not None and ec is not None:
            lat[pname]["n"] += 1
            lat[pname]["wins"] += int(ei > ec)
            lat_logr[pname].append(math.log10((ei + 1e-9) / (ec + 1e-9)))
        if ei is not None and eo is not None:
            seg[ipsi]["n"] += 1
            seg[ipsi]["wins"] += int(ei > eo)
        rows.append({"subject_id": sid, "group": status, "task": w["task"], "side": side,
                     "medication": w["medication"] or "", "start_ms": w["start_ms"], "stop_ms": w["stop_ms"],
                     "pair": pname, "ipsi_sensor": ipsi, "contra_sensor": contra,
                     **{f"energy_{c}": ("" if e[c] is None else round(e[c], 6)) for c in ("lh", "rh", "ll", "rl")},
                     "ipsi_moves_more": "" if (ei is None or ec is None) else int(ei > ec),
                     "log10_ipsi_over_contra": "" if (ei is None or ec is None) else round(math.log10((ei + 1e-9) / (ec + 1e-9)), 3)})
    align = {}
    for code in ("ch", "ll", "rl"):
        if code not in data:
            continue
        ew = [dyn_energy(*data[code], w["start_ms"] + offset_ms, w["stop_ms"] + offset_ms) for w in windows if w["task"] in WALK_TASKS]
        es = [dyn_energy(*data[code], w["start_ms"] + offset_ms, w["stop_ms"] + offset_ms) for w in windows if w["task"] in STILL_TASKS]
        ew, es = [x for x in ew if x is not None], [x for x in es if x is not None]
        if ew and es:
            align[code] = (len(ew), len(es), math.log10((float(np.median(ew)) + 1e-9) / (float(np.median(es)) + 1e-9)))
    ch_al = align.get("ch")
    if ch_al is None:
        verdict = "not_measurable"
    elif ch_al[2] >= ALIGN_MIN_LOG10:
        verdict = "aligned"
    else:
        verdict = "not_aligned_or_no_walking"
    return {"lat": lat, "lat_logr": lat_logr, "seg": seg, "align": align, "align_verdict": verdict, "rows": rows}


def search_offset(ch, windows):
    """One constant offset (s) of the annotation clock onto the chest sensor clock that maximizes
    log10(median walking-window energy / median still-window energy), per-second energies, searched
    from OFFSET_MIN_S to OFFSET_MAX_S. Returns (offset, best log10, margin over the best offset more
    than OFFSET_EXCLUDE_S away)."""
    t, xyz = ch
    ok = np.isfinite(xyz).all(axis=1)
    t, xyz = t[ok], xyz[ok]
    sec = (t // 1000).astype(np.int64)
    nb = int(sec.max()) + 1
    cnt = np.bincount(sec, minlength=nb).astype(float)
    e = np.zeros(nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(3):
            s1 = np.bincount(sec, xyz[:, k], nb)
            s2 = np.bincount(sec, xyz[:, k] ** 2, nb)
            e += s2 / cnt - (s1 / cnt) ** 2
    e = np.nan_to_num(e)
    walk = [(w["start_ms"] // 1000, w["stop_ms"] // 1000) for w in windows if w["task"] in WALK_TASKS]
    still = [(w["start_ms"] // 1000, w["stop_ms"] // 1000) for w in windows if w["task"] in STILL_TASKS]
    if not walk or not still:
        return None, float("nan"), float("nan")
    scores = []
    for off in range(OFFSET_MIN_S, OFFSET_MAX_S + 1, OFFSET_STEP_S):
        def med(ws):
            vals = []
            for a0, b0 in ws:
                a1, b1 = a0 + off, b0 + off
                if a1 < 0 or b1 >= nb:
                    return None
                vals.append(float(np.median(e[a1:b1 + 1])))
            return float(np.median(vals))
        mw, ms = med(walk), med(still)
        if mw is None or ms is None:
            continue
        scores.append((math.log10((mw + 1e-9) / (ms + 1e-9)), off))
    if not scores:
        return None, float("nan"), float("nan")
    best, off = max(scores)
    rest = [s for s, o in scores if abs(o - off) > OFFSET_EXCLUDE_S]
    return off, best, best - (max(rest) if rest else float("nan"))


def biostamp_docs(a, clinic):
    """What the MoveAtlas documents say about the group split and the placement, read as text (the
    ingester is read, never run or imported), against the clinic CSV."""
    root = f"{a.moveatlas}/datasets/imu/pd_biostamprc21"
    doc = {"inputs": []}
    p = f"{root}/datacard.md"
    if os.path.isfile(p):
        doc["inputs"].append(p)
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^- population:\s*(.*)$", line.strip())
                if m:
                    doc["datacard_population"] = m.group(1).strip()
    p = f"{a.moveatlas}/docs/datasets_index.csv"
    if os.path.isfile(p):
        doc["inputs"].append(p)
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["id"] == "pd_biostamprc21":
                    doc["registry_n_subjects"] = r["n_subjects"]
    p = f"{a.moveatlas}/docs/demographics_curated.csv"
    if os.path.isfile(p):
        doc["inputs"].append(p)
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["dataset"] == "pd_biostamprc21":
                    doc["demographics_curated"] = {k: r[k] for k in ("n_participants", "age_statement", "sex_statement", "source_url")}
    p = f"{a.moveatlas}/ingest_pd_biostamprc21.py"
    if os.path.isfile(p):
        doc["inputs"].append(p)
        with open(p, encoding="utf-8") as f:
            src = f.read()
        m = re.search(r"(\d+) PD \+ (\d+) controls", src)
        doc["ingester_docstring_groups"] = m.group(0) if m else ""
        ing = dict((c, (slot, pl)) for c, slot, pl in re.findall(r'"(ch|ll|rl|lh|rh)":\s*\("([^"]+)",\s*"([^"]+)"\)', src))
        doc["ingester_sensors_table"] = {c: {"slot": s, "placement": pl} for c, (s, pl) in sorted(ing.items())}
        doc["placement_constant_matches_ingester"] = int(
            len(ing) == len(BIOSTAMP_SENSORS)
            and all(c in ing and ing[c][0] == slot and ing[c][1].startswith(pl) for c, (pl, slot, _) in BIOSTAMP_SENSORS.items()))
    # the clinic CSV against the curated demographics
    by = collections.defaultdict(list)
    for r in clinic:
        by[r.get("Status", "").strip()].append(r)
    doc["clinic_by_group"] = {}
    for g, rs in sorted(by.items()):
        ages = np.array([float(r["Age"]) for r in rs if r.get("Age", "").strip()])
        women = sum(1 for r in rs if r.get("Sex", "").strip().lower() == "female")
        doc["clinic_by_group"][g] = {"n": len(rs), "age_mean": round(float(ages.mean()), 1),
                                     "age_sd_ddof1": round(float(ages.std(ddof=1)), 1),
                                     "n_women": women, "pct_women": round(100.0 * women / len(rs), 1)}
    return doc


def run_biostamp(a, out):
    root = f"{a.moveatlas}/datasets/imu/pd_biostamprc21"
    clinic_path = f"{root}/raw/Clinic_DataPDBioStampRCStudy.csv"
    zpath = f"{root}/raw/FullDataSet_PD-BioStampRC21.zip"
    czip = f"{root}/raw/Clinic_Data_PD-BioStampRC21.zip"
    audit_path = f"{root}/processed/biostamp_ingest_audit.csv"
    with open(clinic_path, encoding="utf-8", newline="") as f:
        clinic = list(csv.DictReader(f))
    with open(clinic_path, "rb") as f:
        clinic_bytes = f.read()
    with zipfile.ZipFile(czip) as cz:
        inner = [n for n in cz.namelist() if n.endswith(".csv")]
        clinic_zip_same = int(len(inner) == 1 and cz.read(inner[0]) == clinic_bytes)
    audit = {}
    if os.path.isfile(audit_path):
        with open(audit_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                audit[(r["subject"], r["sensor"])] = r
    zf = zipfile.ZipFile(zpath)
    names = set(zf.namelist())
    subjects = sorted({n.split("/")[0] for n in names if re.match(r"^\d{3}/", n)})
    clinic_by_id = {r["ID"].strip(): r for r in clinic}
    group_rows, place_rows, win_rows, lat_rows = [], [], [], []
    lat_totals = collections.Counter()
    ann_totals = collections.Counter()
    for si, sid in enumerate(subjects, 1):
        c = clinic_by_id.get(sid, {})
        status = c.get("Status", "").strip()
        base_cols = [k for k in c if re.match(r"updrs_3_17[a-e]$", k)]
        off_cols = [k for k in c if k.endswith("_off")]
        on_cols = [k for k in c if k.endswith("_on")]
        filled = lambda cols: sum(1 for k in cols if c.get(k, "").strip() != "")  # noqa: E731
        clinic_pattern = ("on_off_columns" if filled(off_cols) and filled(on_cols)
                          else "single_columns" if filled(base_cols) else "none")
        ann = f"{sid}/AnnotID{sid}.csv"
        windows, ann_stats = [], collections.Counter()
        if ann in names:
            rows = list(csv.reader(io.TextIOWrapper(zf.open(ann), encoding="utf-8", errors="replace")))
            windows, ann_stats = parse_annotations(rows[1:])
        ann_totals.update(ann_stats)
        med = collections.Counter((w["medication"] or "") if w["medication"] is not None else "no_row" for w in windows)
        on_w = [w for w in windows if w["medication"] == "ON"]
        off_w = [w for w in windows if w["medication"] == "OFF"]
        upd_on = sum(1 for w in on_w if w["task"].startswith("UPDRS 3."))
        upd_off = sum(1 for w in off_w if w["task"].startswith("UPDRS 3."))
        order = ""
        if on_w and off_w:
            if max(w["stop_ms"] for w in off_w) <= min(w["start_ms"] for w in on_w):
                order = "OFF_then_ON"
            elif max(w["stop_ms"] for w in on_w) <= min(w["start_ms"] for w in off_w):
                order = "ON_then_OFF"
            else:
                order = "interleaved"
        if on_w and off_w and upd_on and upd_off:
            med_class = "tagged_ON_and_OFF"
        elif on_w or off_w:
            med_class = "stray_tags_only"
        else:
            med_class = "untagged"
        n_uhdrs = sum(1 for w in windows if w["task"].startswith("UHDRS"))
        n_updrs3 = sum(1 for w in windows if w["task"].startswith("UPDRS 3."))
        max_ms = max((w["stop_ms"] for w in windows), default=0)
        # sensors and the measured placement check
        data = {}
        for code in BIOSTAMP_SENSORS:
            fn = f"{sid}/{code}_ID{sid}Accel.csv"
            if fn in names and max_ms > 0:
                data[code] = read_sensor_head(zf, fn, max_ms + 60000)
        res0 = measure_placement(sid, status, windows, data, 0)
        res, offset_s, offset_best, offset_margin = res0, "", "", ""
        if res0["align_verdict"] != "aligned" and "ch" in data:
            # the annotation windows do not line up with the movement: search one constant clock offset
            data = {}
            for code in BIOSTAMP_SENSORS:
                fn = f"{sid}/{code}_ID{sid}Accel.csv"
                if fn in names:
                    data[code] = read_sensor_head(zf, fn, max_ms + OFFSET_MAX_S * 1000 + 60000)
            off, best, margin = search_offset(data["ch"], windows)
            offset_best, offset_margin = round(best, 3), round(margin, 3)
            if off is not None and best >= ALIGN_MIN_LOG10:
                offset_s = off
                res = measure_placement(sid, status, windows, data, off * 1000)
        for r in res["rows"]:
            r["clock_offset_applied_s"] = offset_s
        lat_rows.extend(res["rows"])
        lat, lat_logr, seg, align = res["lat"], res["lat_logr"], res["seg"], res["align"]
        ch_al, ch_al0 = align.get("ch"), res0["align"].get("ch")
        align_verdict = res0["align_verdict"]
        if align_verdict != "aligned" and offset_s != "":
            align_verdict = "aligned_after_offset" if res["align_verdict"] == "aligned" else "not_aligned"

        def verdict_of(n, wins):
            if n < LAT_MIN_WINDOWS:
                return "too_few_windows"
            if wins / n >= LAT_CONSISTENT:
                return "consistent"
            if wins / n <= 1 - LAT_CONSISTENT:
                return "reversed"
            return "inconclusive"

        for code, (placement, slot, proxy) in BIOSTAMP_SENSORS.items():
            fn = f"{sid}/{code}_ID{sid}Accel.csv"
            present = fn in names
            au = audit.get((sid, code), {})
            pname = {"lh": "forearms", "rh": "forearms", "ll": "thighs", "rl": "thighs"}.get(code)
            n = lat[pname]["n"] if pname else 0
            wins = lat[pname]["wins"] if pname else 0
            if code == "ch":
                verdict = "not_tested"
            elif not present:
                verdict = "file_missing"
            else:
                verdict = verdict_of(n, wins)
            verdict0 = ("not_tested" if code == "ch" else "file_missing" if not present
                        else verdict_of(res0["lat"][pname]["n"], res0["lat"][pname]["wins"]))
            lat_totals[(code, verdict)] += 1
            sn, sw = seg[code]["n"], seg[code]["wins"]
            first_ts = ""
            if present:
                first_ts = int(data[code][0][0]) if code in data and len(data[code][0]) else ""
            place_rows.append({
                "subject_id": sid, "subject_ns": f"pd_biostamprc21:{sid}", "group": status, "sensor_code": code,
                "file_present": int(present), "file_name": fn if present else "",
                "uncompressed_bytes": zf.getinfo(fn).file_size if present else "",
                "documented_placement": placement, "moveatlas_slot": slot, "moveatlas_slot_is_proxy": proxy,
                "placement_source": PLACEMENT_SOURCE,
                "first_timestamp_ms": first_ts,
                "ingest_audit_n_samples": au.get("n_samples", ""), "ingest_audit_fs_hz": au.get("fs_measured", ""),
                "ingest_audit_median_acc_norm_g": au.get("median_acc_norm_g", ""),
                "lateral_pair": pname or "", "lateral_pair_windows": n if code != "ch" else "",
                "lateral_pair_ipsi_moves_more": wins if code != "ch" else "",
                "lateral_pair_median_log10_ratio": (round(float(np.median(lat_logr[pname])), 3) if pname and lat_logr[pname] else ""),
                "lateral_check": verdict,
                "lateral_check_at_zero_offset": verdict0,
                "clock_offset_applied_s": offset_s,
                "segment_windows": sn if code != "ch" else "", "segment_this_moves_more": sw if code != "ch" else "",
                "segment_check": ("not_tested" if code == "ch" else "too_few_windows" if sn < LAT_MIN_WINDOWS
                                  else "consistent" if sw / sn >= LAT_CONSISTENT
                                  else "reversed" if sw / sn <= 1 - LAT_CONSISTENT else "inconclusive"),
            })
        span = lambda ws, k, f: round(f(w[k] for w in ws) / 1000.0, 1) if ws else ""  # noqa: E731
        group_rows.append({
            "subject_id": sid, "subject_ns": f"pd_biostamprc21:{sid}", "group": status,
            "sex": c.get("Sex", ""), "age": c.get("Age", ""),
            "clinic_updrs_3_17_pattern": clinic_pattern,
            "annotation_file": int(ann in names), "n_windows": len(windows),
            "n_windows_OFF": med["OFF"], "n_windows_ON": med["ON"], "n_windows_untagged": med[""] + med["no_row"],
            "n_updrs3_windows": n_updrs3, "n_updrs3_OFF": upd_off, "n_updrs3_ON": upd_on, "n_uhdrs_windows": n_uhdrs,
            "medication_windows": med_class, "session_order": order,
            "OFF_first_start_s": span(off_w, "start_ms", min), "OFF_last_stop_s": span(off_w, "stop_ms", max),
            "ON_first_start_s": span(on_w, "start_ms", min), "ON_last_stop_s": span(on_w, "stop_ms", max),
            "last_annotation_stop_s": round(max_ms / 1000.0, 1),
            "align_walk_windows": ch_al0[0] if ch_al0 else "", "align_still_windows": ch_al0[1] if ch_al0 else "",
            "align_chest_log10_walk_over_still": round(ch_al0[2], 3) if ch_al0 else "",
            "clock_offset_search_best_log10": offset_best, "clock_offset_search_margin_log10": offset_margin,
            "clock_offset_applied_s": offset_s,
            "align_chest_log10_after_offset": round(ch_al[2], 3) if (ch_al and offset_s != "") else "",
            "align_thigh_l_log10_walk_over_still": round(align["ll"][2], 3) if "ll" in align else "",
            "align_thigh_r_log10_walk_over_still": round(align["rl"][2], 3) if "rl" in align else "",
            "annotation_sensor_alignment": align_verdict,
            "annotation_task_rows": ann_stats["task_rows"],
            "annotation_duplicate_task_rows": ann_stats["duplicate_task_rows"],
            "annotation_attribute_rows": ann_stats["attribute_rows"],
            "attribute_rows_out_of_sequence": ann_stats["attribute_rows_out_of_sequence"],
            "attribute_rows_unmatched_fallback": ann_stats["attribute_rows_unmatched_fallback"],
            "attribute_rows_ambiguous_stop": ann_stats["attribute_rows_ambiguous_stop"],
            "attribute_conflicts": ann_stats["attribute_conflicts"],
            "orphan_attribute_rows": ann_stats["orphan_attribute_rows"],
            "n_sensor_files": sum(1 for code in BIOSTAMP_SENSORS if f"{sid}/{code}_ID{sid}Accel.csv" in names),
            "sensors_missing": ";".join(code for code in BIOSTAMP_SENSORS if f"{sid}/{code}_ID{sid}Accel.csv" not in names),
        })
        for w in windows:
            win_rows.append({"subject_id": sid, "group": status, "task": w["task"], "start_ms": w["start_ms"],
                             "stop_ms": w["stop_ms"], "duration_s": round((w["stop_ms"] - w["start_ms"]) / 1000.0, 3),
                             "side": window_side(w), "area": w["attrs"].get("area of the body", ""),
                             "medication": w["medication"] if w["medication"] else ("untagged" if w["medication"] == "" else "no_medication_row"),
                             "duplicate_rows_collapsed": w["duplicate_rows"]})
        print(f"  pd_biostamprc21 [{si:2d}/{len(subjects)}] {sid} {status}: {len(windows)} windows, "
              f"{med_class}, sensors read {len(data)}", flush=True)
        del data
    write_csv(os.path.join(out, "pd_biostamprc21_groups.csv"), group_rows)
    write_csv(os.path.join(out, "pd_biostamprc21_placement.csv"), place_rows)
    write_csv(os.path.join(out, "pd_biostamprc21_med_windows.csv"), win_rows)
    write_csv(os.path.join(out, "pd_biostamprc21_placement_check_windows.csv"), lat_rows)
    g = pd.DataFrame(group_rows)
    p = pd.DataFrame(place_rows)
    # subjects whose pair check does not come out consistent
    pair_view = (p[p.lateral_pair != ""].drop_duplicates(["subject_id", "lateral_pair"])
                 [["subject_id", "group", "lateral_pair", "lateral_pair_windows", "lateral_pair_ipsi_moves_more",
                   "lateral_pair_median_log10_ratio", "lateral_check"]])
    pair_counts = collections.Counter(f"{r.lateral_pair}:{r.lateral_check}" for r in pair_view.itertuples())
    not_consistent = pair_view[~pair_view.lateral_check.isin(["consistent"])].to_dict("records")
    zero_first = collections.Counter()
    zero_per_subject = collections.Counter()
    for r in p.itertuples():
        if str(r.first_timestamp_ms) == "0":
            zero_first[r.sensor_code] += 1
            zero_per_subject[r.subject_id] += 1
    docs = biostamp_docs(a, clinic)
    summ = {
        "n_subjects_zip": len(subjects), "n_subjects_clinic": len(clinic),
        "clinic_zip_copy_identical": clinic_zip_same,
        "moveatlas_documents": docs,
        "group_counts": g["group"].value_counts().to_dict(),
        "sex_by_group": {k: v.value_counts().to_dict() for k, v in g.groupby("group")["sex"]},
        "age_median_by_group": {k: float(pd.to_numeric(v).median()) for k, v in g.groupby("group")["age"]},
        "clinic_pattern_by_group": {k: v.value_counts().to_dict() for k, v in g.groupby("group")["clinic_updrs_3_17_pattern"]},
        "medication_windows_by_group": {k: v.value_counts().to_dict() for k, v in g.groupby("group")["medication_windows"]},
        "session_order_tagged": g[g.medication_windows == "tagged_ON_and_OFF"]["session_order"].value_counts().to_dict(),
        "stray_tag_subjects": g[g.medication_windows == "stray_tags_only"][["subject_id", "group", "n_windows_OFF", "n_windows_ON"]].to_dict("records"),
        "tagged_subjects_not_on_off_in_clinic": g[(g.medication_windows == "tagged_ON_and_OFF") & (g.clinic_updrs_3_17_pattern != "on_off_columns")]["subject_id"].tolist(),
        "on_off_clinic_subjects_not_tagged": g[(g.clinic_updrs_3_17_pattern == "on_off_columns") & (g.medication_windows != "tagged_ON_and_OFF")]["subject_id"].tolist(),
        "n_windows_total": len(win_rows),
        "annotation_row_counts": {k: int(ann_totals[k]) for k in (
            "task_rows", "duplicate_task_rows", "attribute_rows", "attribute_rows_out_of_sequence",
            "attribute_rows_unmatched_fallback", "attribute_rows_ambiguous_stop", "attribute_conflicts",
            "orphan_attribute_rows", "short_rows")},
        "subjects_with_attribute_rows_out_of_sequence": int((g.attribute_rows_out_of_sequence > 0).sum()),
        "annotation_rule": "an attribute row (Medication, Hand, Side, Foot, Leg, Arm, Direction, Area of the body) is "
                           "attached to the task window whose stop timestamp equals its own timestamp, not to the "
                           "preceding task row; identical task rows (same type, start, stop) are one window",
        "subjects_with_uhdrs_windows": g[g.n_uhdrs_windows > 0][["subject_id", "group", "n_uhdrs_windows"]].to_dict("records"),
        "sensor_files_present": int(p.file_present.sum()), "sensor_files_expected": len(p),
        "missing_sensor_files": p[p.file_present == 0][["subject_id", "sensor_code"]].to_dict("records"),
        "alignment_counts": g["annotation_sensor_alignment"].value_counts().to_dict(),
        "alignment_rule": f"chest dynamic energy, median over walking windows {sorted(WALK_TASKS)} against median over "
                          f"seated still windows {sorted(STILL_TASKS)}; aligned when log10 of the ratio >= {ALIGN_MIN_LOG10}",
        "alignment_not_aligned": g[g.annotation_sensor_alignment != "aligned"][
            ["subject_id", "group", "align_walk_windows", "align_still_windows", "align_chest_log10_walk_over_still",
             "clock_offset_search_best_log10", "clock_offset_search_margin_log10", "clock_offset_applied_s",
             "align_chest_log10_after_offset", "annotation_sensor_alignment"]].to_dict("records"),
        "offset_search": f"constant offset searched from {OFFSET_MIN_S} s to {OFFSET_MAX_S} s in {OFFSET_STEP_S} s steps on "
                         f"per-second chest energies; applied when the best log10 walk over still reaches {ALIGN_MIN_LOG10}; "
                         f"margin = best minus the best offset more than {OFFSET_EXCLUDE_S} s away",
        "alignment_chest_log10_min_median_max": [round(float(pd.to_numeric(g.align_chest_log10_walk_over_still, errors="coerce").min()), 3),
                                                 round(float(pd.to_numeric(g.align_chest_log10_walk_over_still, errors="coerce").median()), 3),
                                                 round(float(pd.to_numeric(g.align_chest_log10_walk_over_still, errors="coerce").max()), 3)],
        "lateral_pair_check_counts": dict(sorted(pair_counts.items())),
        "lateral_pair_not_consistent": not_consistent,
        "n_lateral_windows": len(lat_rows),
        "lateral_rule": f"per subject and sensor pair (forearms lh/rh on the hand items {sorted(HAND_TASKS)}, thighs ll/rl on "
                        f"the leg items {sorted(LEG_TASKS)}): the share of windows in which the sensor on the tested side "
                        f"has a larger dynamic energy (sum of per-axis variances of the acceleration inside the window) "
                        f"than the opposite one; consistent at >= {LAT_CONSISTENT} with at least {LAT_MIN_WINDOWS} "
                        f"windows, reversed at <= {round(1 - LAT_CONSISTENT, 2)}, inconclusive between",
        "clock_assumption": "annotation and sensor timestamps are read as one session clock in ms (each sensor CSV starts at "
                            "its own offset on that clock and exactly one sensor per subject starts at 0)",
        "first_timestamp_zero_sensor_counts": dict(zero_first),
        "subjects_by_number_of_sensors_starting_at_0": dict(collections.Counter(zero_per_subject.values())),
        "inputs": [clinic_path, czip, zpath, audit_path] + docs["inputs"],
    }
    sc = collections.Counter(f"{r.sensor_code}:{r.segment_check}" for r in p.itertuples())
    summ["segment_check_counts"] = dict(sorted(sc.items()))
    print(f"pd_biostamprc21: {len(subjects)} subjects; groups {summ['group_counts']}")
    return summ


# ---------------------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


def sha256_small(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x):
    return f"{x:,.1f}" if isinstance(x, float) else str(x)


def write_summary_md(out, head, sw, sc, sb):
    L = []
    L.append("# Medication and group tables (T0.7b)")
    L.append("")
    L.append(f"Generated by scripts/probe_medication.py against MoveAtlas {head}. Every number below is read "
             "from medication_summary.json, written by the same run. The tables sit beside this file.")
    L.append("")
    L.append("## weargait_pd: levodopa-equivalent daily dose")
    L.append("")
    st = sw["status_counts"]
    L.append(f"- PD subjects in the raw table: {sw['n_pd_subjects']} ({', '.join(f'{k} {v}' for k, v in sorted(sw['n_by_id_prefix'].items()))}).")
    L.append("- Status per subject: " + ", ".join(f"{k} {st[k]}" for k in sorted(st)) + ".")
    L.append("  legible: every dosed PD entry parsed with no review flag; legible_review: parsed, with at least one "
             "rule decision to read (expired entry kept, conflicting duplicates, q-hours schedule, titration or "
             "replacement note, co-dose note); partial: at least one LEDD drug named without a parsable dose "
             "(the LEDD column is empty and ledd_lower_bound_mg holds the sum of the parsed part); illegible: PD "
             "drugs named, none dosed; no_pd_drug_listed: a medication list without any PD drug; missing: both "
             "medication fields empty.")
    for p_, d in sorted(sw["status_counts_by_prefix"].items()):
        L.append(f"- {p_}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())) + ".")
    if sw["ledd_legible_median_mg"] is not None:
        L.append(f"- LEDD over the {sw['n_legible_total']} legible subjects (legible and legible_review): median "
                 f"{sw['ledd_legible_median_mg']} mg, interquartile range {sw['ledd_legible_iqr_mg'][0]} to "
                 f"{sw['ledd_legible_iqr_mg'][1]} mg, range {sw['ledd_legible_min_max_mg'][0]} to {sw['ledd_legible_min_max_mg'][1]} mg.")
    unv = [(k, v) for k, v in LEDD_FACTORS.items() if v[4] == "extension_unverified"]
    L.append(f"- {sw['n_legible_using_unverified_factor']} legible subjects use at least one factor that Tomlinson 2010 "
             f"does not cover (counts: {', '.join(f'{k} {v}' for k, v in sorted(sw['unverified_factor_use_counts'].items()))}). "
             f"Those factors ({'; '.join(f'{k} {v[2]} ({v[1]})' for k, v in unv)}) are recalled from Jost et al. 2023 "
             "and could not be checked on this machine (no copy of the paper on disk, no network); the raw levodopa mg "
             "per formulation are in the table so a corrected factor is a recomputation.")
    L.append(f"- Subjects whose LEDD sums two or more strengths of the same levodopa formulation (info flag "
             f"levodopa_multiple_strengths_summed, read literally from the list): {sw['n_multiple_strengths_summed']} "
             f"({', '.join(sw['subjects_multiple_strengths_summed'])}). Largest raw levodopa mg/day of one formulation: "
             f"{sw['max_raw_levodopa_one_formulation']['mg_day']} mg ({sw['max_raw_levodopa_one_formulation']['subject_id']}, "
             f"{sw['max_raw_levodopa_one_formulation']['drug']}). An entry counted as 0 mg/day (PRN only) does not "
             f"count as a summed strength. Other drugs summed over two or more strengths (info flag "
             f"other_drug_multiple_strengths_summed): "
             + (", ".join(f"{k} ({v})" for k, v in sorted(sw['subjects_other_drug_multiple_strengths_summed'].items()))
                or "none") + ".")
    L.append(f"- DBS field: {', '.join(f'{k} {v}' for k, v in sorted(sw['dbs_counts'].items()))} "
             f"(targets among Yes: {', '.join(f'{k} {v}' for k, v in sorted(sw['dbs_target_counts'].items()))}; "
             f"laterality: {', '.join(f'{k} {v}' for k, v in sorted(sw['dbs_laterality_counts'].items()))}).")
    L.append(f"- Subjects with DBS = No: {sw['n_no_dbs']}; of them legible: {sw['n_no_dbs_legible']} "
             f"(status: {', '.join(f'{k} {v}' for k, v in sorted(sw['n_no_dbs_status'].items()))}).")
    L.append(f"- Time from last dose to session, same-day reading: median {sw['hours_since_last_dose_median']} h over "
             f"{sw['n_hours_since_last_dose_clean']} subjects without a time flag; flags: "
             f"{', '.join((k or 'none') + ' ' + str(v) for k, v in sorted(sw['dose_time_flags'].items()))}.")
    L.append(f"- PD drug entries parsed: {sw['n_pd_entries_parsed']}; dropped by rule: "
             f"{', '.join(f'{k} {v}' for k, v in sorted(sw['entry_drop_reasons'].items()))}.")
    L.append(f"- Review flags (subjects carrying each): {', '.join(f'{k} {v}' for k, v in sorted(sw['review_flag_subject_counts'].items()))}.")
    for s_, ids in sorted(sw["subjects_by_status_not_legible"].items()):
        L.append(f"- Subjects with status {s_}: {', '.join(ids)}.")
    if sw["subjects_with_unrecognized_free_text_items"]:
        L.append("- Unrecognized items in free-text medication fields (possibly PD drugs the lexicon does not know): "
                 + "; ".join(f"{k}: {v}" for k, v in sorted(sw["subjects_with_unrecognized_free_text_items"].items())) + ".")
    L.append(f"- Controls: {sw['n_controls']}; controls whose medication column names a PD drug: "
             f"{len(sw['controls_with_pd_drug_named'])}"
             + (f" ({'; '.join(d['subject_id'] + ': ' + ','.join(d['drugs']) for d in sw['controls_with_pd_drug_named'])})" if sw['controls_with_pd_drug_named'] else "") + ".")
    L.append("- Files: ledd_weargait_pd.csv (one row per PD subject), ledd_weargait_pd_entries.csv (one row per PD "
             "drug entry with its parse and rule decisions), ledd_factors.csv (the factor table as used).")
    L.append("")
    L.append("## care_pd: medication state from the raw walk names (evaluation-only data)")
    L.append("")
    L.append(f"- {sc['n_walks']} walks of {sc['n_subjects']} subjects read from the {len(sc['per_cohort'])} raw "
             f"cohort pickles; walks not in the "
             f"MoveAtlas processed manifest: {sc['raw_walks_not_in_manifest']}; manifest walks not in the raw pickles: "
             f"{sc['manifest_walks_not_in_raw']}.")
    for c, pc in sc["per_cohort"].items():
        line = (f"- {c}: {pc['n_subjects']} subjects, {pc['n_walks']} walks; walks OFF {pc['walks_OFF']}, ON "
                f"{pc['walks_ON']}, no state in the name {pc['walks_no_state']}; subjects with OFF and ON walks "
                f"{pc['subjects_with_OFF_and_ON']}; walks whose record carries a medication field "
                f"{pc['walks_medication_field_set']}; name against medication field disagreements "
                f"{pc['walks_name_field_disagree']}; non-standard names parsed {pc['walks_nonstandard_name_parsed']}")
        if pc["name_tokens"]:
            line += "; name tokens " + ", ".join(f"{k} {v}" for k, v in sorted(pc["name_tokens"].items()))
        if "subjects_with_OFF_and_ON_by_group" in pc and pc["subjects_with_OFF_and_ON"]:
            line += "; subjects with both states by group field: " + ", ".join(
                f"{k} {v}" for k, v in sorted(pc["subjects_with_OFF_and_ON_by_group"].items()))
        L.append(line + ".")
    L.append("- Files: care_pd_med_state.csv (one row per walk), care_pd_med_state_subjects.csv (one row per subject).")
    L.append("")
    L.append("## pd_biostamprc21: group, placement and medication windows")
    L.append("")
    dm = sb["moveatlas_documents"]
    cur = dm.get("demographics_curated", {})
    L.append(f"- Subjects: {sb['n_subjects_zip']} in the raw zip, {sb['n_subjects_clinic']} in the clinic CSV; groups "
             f"from the clinic Status column: {', '.join(f'{k} {v}' for k, v in sorted(sb['group_counts'].items()))}. "
             f"The clinic CSV inside Clinic_Data_PD-BioStampRC21.zip is byte-identical to the loose copy: "
             f"{'yes' if sb['clinic_zip_copy_identical'] else 'no'}.")
    pop = dm.get("datacard_population", "")
    datacard_disagrees = bool(pop) and "control" not in pop.lower() and len(sb["group_counts"]) > 1
    line = (f"- MoveAtlas documents on the group split: the datacard population line reads "
            f"\"{pop or '(not found)'}\"; the registry row gives n_subjects "
            f"{dm.get('registry_n_subjects', '(not found)')} without a split; docs/demographics_curated.csv gives "
            f"\"{cur.get('n_participants', '(not found)')}\" with age \"{cur.get('age_statement', '')}\" and sex "
            f"\"{cur.get('sex_statement', '')}\"; the ingester docstring says \"{dm.get('ingester_docstring_groups', '')}\". "
            f"Recomputed from the clinic CSV: "
            + "; ".join(f"{g} n {v['n']}, age {v['age_mean']} [{v['age_sd_ddof1']}] (mean [SD]), women {v['n_women']} "
                        f"({v['pct_women']}%)" for g, v in dm.get("clinic_by_group", {}).items()) + ".")
    if datacard_disagrees:
        line += (" The datacard population line does not match the clinic CSV; that is a MoveAtlas correction for "
                 "the owner, not applied here.")
    L.append(line)
    arc = sb["annotation_row_counts"]
    L.append(f"- Annotation files: {arc['task_rows']} task rows, of which {arc['duplicate_task_rows']} repeat an "
             f"identical row (same task, start and stop) and are collapsed, giving {sb['n_windows_total']} windows; "
             f"{arc['attribute_rows']} attribute rows. Rule: {sb['annotation_rule']}. Attribute rows that sit out of "
             f"sequence (their window is not the preceding task row): {arc['attribute_rows_out_of_sequence']}, in "
             f"{sb['subjects_with_attribute_rows_out_of_sequence']} subjects; a reader that attaches attributes to the "
             f"preceding task row gives these to the wrong window. Rows matching no task stop (fallback to the "
             f"preceding task row): {arc['attribute_rows_unmatched_fallback']}; rows whose stop is shared by two "
             f"distinct windows: {arc['attribute_rows_ambiguous_stop']}; windows with two different values of one "
             f"attribute: {arc['attribute_conflicts']}; orphan rows: {arc['orphan_attribute_rows']}.")
    L.append(f"- Medication windows by group: {json.dumps(sb['medication_windows_by_group'])}; session order of the "
             f"tagged subjects: {json.dumps(sb['session_order_tagged'])}.")
    L.append(f"- Subjects with stray medication tags only: {json.dumps(sb['stray_tag_subjects'])}.")
    L.append(f"- Clinic UPDRS 3.17 column pattern by group: {json.dumps(sb['clinic_pattern_by_group'])}; tagged subjects "
             f"without ON and OFF clinic columns: {sb['tagged_subjects_not_on_off_in_clinic']}; subjects with ON and OFF "
             f"clinic columns but no tagged windows: {sb['on_off_clinic_subjects_not_tagged']}.")
    L.append(f"- Subjects with UHDRS item windows: {json.dumps(sb['subjects_with_uhdrs_windows'])}.")
    L.append(f"- Sensor files present: {sb['sensor_files_present']} of {sb['sensor_files_expected']}; missing: "
             f"{json.dumps(sb['missing_sensor_files'])}.")
    L.append("- Placement per sensor code (documented, not measured): "
             + ", ".join(f"{c} {pl} (MoveAtlas slot {slot}{', proxy' if proxy else ''})"
                         for c, (pl, slot, proxy) in BIOSTAMP_SENSORS.items())
             + f". Source: {PLACEMENT_SOURCE}. The constant in this script matches the ingester's SENSORS table read "
               f"as text: {'yes' if dm.get('placement_constant_matches_ingester') else 'no'}. The raw zip carries no "
               "README; its file names carry only the codes.")
    L.append(f"- Annotation clock against sensor clock: {json.dumps(sb['alignment_counts'])}; chest log10 walk over still, "
             f"min, median, max: {sb['alignment_chest_log10_min_median_max']}. Rule: {sb['alignment_rule']}. Not aligned "
             f"at zero offset: {json.dumps(sb['alignment_not_aligned'])}. Offset search: {sb['offset_search']}; the "
             f"placement verdicts of a subject re-aligned this way are measured on the shifted windows "
             f"(lateral_check_at_zero_offset keeps the unshifted verdict).")
    L.append(f"- Measured left/right check per sensor pair over {sb['n_lateral_windows']} lateralized windows: "
             f"{json.dumps(sb['lateral_pair_check_counts'])}. Rule: {sb['lateral_rule']}.")
    L.append(f"- Pairs not consistent: {json.dumps(sb['lateral_pair_not_consistent'])}.")
    L.append(f"- Measured segment check per sensor (forearm moves more than the same-side thigh in hand items; thigh more "
             f"than the same-side forearm in leg items): {json.dumps(sb['segment_check_counts'])}.")
    L.append(f"- Clock: {sb['clock_assumption']}; sensors whose first timestamp is 0: "
             f"{json.dumps(sb['first_timestamp_zero_sensor_counts'])}; subjects by number of sensors starting at 0: "
             f"{json.dumps(sb['subjects_by_number_of_sensors_starting_at_0'])}.")
    L.append("- Files: pd_biostamprc21_groups.csv (one row per subject), pd_biostamprc21_placement.csv (one row per "
             "subject and sensor), pd_biostamprc21_med_windows.csv (one row per annotated window), "
             "pd_biostamprc21_placement_check_windows.csv (one row per lateralized window with the four limb energies).")
    L.append("")
    text = "\n".join(L) + "\n"
    if "\u2014" in text:
        text = text.replace("\u2014", ", ")
    with open(os.path.join(out, "medication_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    a = parse_args("T0.7b: medication and group tables for weargait_pd, care_pd and pd_biostamprc21")
    out = a.out
    os.makedirs(out, exist_ok=True)
    head = moveatlas_head(a.moveatlas)
    print("MoveAtlas:", head)
    sw = run_weargait(a, out)
    sc = run_care_pd(a, out)
    sb = run_biostamp(a, out)
    summary = {"weargait_pd": sw, "care_pd": sc, "pd_biostamprc21": sb,
               "ledd_factors": {k: {"factor": v[2], "kind": v[1], "status": v[4]} for k, v in LEDD_FACTORS.items()}}
    with open(os.path.join(out, "medication_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
    write_summary_md(out, head, sw, sc, sb)
    small = sw["inputs"] + [sb["inputs"][0], sb["inputs"][1], sb["inputs"][3]] + sb["inputs"][4:] + [sc["inputs"][-1]]
    prov = {"script": os.path.basename(sys.argv[0]), "task": "T0.7b", "moveatlas_root": a.moveatlas,
            "moveatlas_head": head, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "runtime_s": round(time.time() - T_START, 1),
            "inputs_sha256": {p: sha256_small(p) for p in small if os.path.isfile(p)},
            "inputs_size_bytes": {p: os.path.getsize(p) for p in sb["inputs"] + sc["inputs"] if os.path.isfile(p)},
            "outputs": sorted(["ledd_weargait_pd.csv", "ledd_weargait_pd_entries.csv", "ledd_factors.csv",
                               "care_pd_med_state.csv", "care_pd_med_state_subjects.csv", "pd_biostamprc21_groups.csv",
                               "pd_biostamprc21_placement.csv", "pd_biostamprc21_med_windows.csv",
                               "pd_biostamprc21_placement_check_windows.csv",
                               "medication_summary.json", "medication_summary.md"]),
            "notes": "reads MoveAtlas read-only; care_pd pickles loaded with a restricted unpickler (numpy arrays and "
                     "plain containers only); pd_biostamprc21 accelerometer CSVs streamed from the zip up to the last "
                     "annotated window plus 60 s (plus OFFSET_MAX_S for a subject whose windows fail the clock "
                     "alignment check); the MoveAtlas ingester is read as text for its placement table, never run or "
                     "imported"}
    with open(os.path.join(out, "_provenance_probe_medication.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print(f"done in {time.time() - T_START:.1f} s")


if __name__ == "__main__":
    main()
