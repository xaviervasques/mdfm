"""Gene class or its surgical correlates: isolated or combined class of each pallidal_units_dystonia gene,
from propagated, frequency-weighted, disease-level HPO annotations.

Task T12 of the weekend plan 2026-10-01; design note docs/MFM_DESIGN_NOTE_CLINICAL_QUESTIONS_2026-09-29.md,
priority 7 (a) and (b) and question 18. Status DRAFT, pending the neurologist.

Rule: reports/gene_classes/rule.txt (items G1 to G16), written and hashed in rule.sha256 before this script
read any annotation row of a cohort gene. assert_rule_hash() stops the run before any input is read when the
SHA-256 of rule.txt differs from RULE_SHA256 or from rule.sha256.

Inputs (read-only, unmodified): hp.obo, phenotype.hpoa.gz and genes_to_phenotype.txt of
datasets/ontology/hpo/raw in MoveAtlas; reports/pallidal_units/patients_per_gene.csv (per-gene counts written
by scripts/probe_pallidal_units.py from identifier columns). No row of the pallidal CSV is read (G15).

Outputs (reports/gene_classes/): entities.csv, classes.csv, summary.md, _provenance.json. Only
_provenance.json carries a time stamp, so two runs give identical bytes for the other three. entities.csv
also carries, for reading only, n_rows_<category> (kept rows under the roots of each category) and
inheritance (the aspect I term names of the entity); neither enters a class. summary.md cites the MoveAtlas
HEAD and diff hash of the stamp.

Run:
  CUDA_VISIBLE_DEVICES=-1 python scripts/build_gene_classes.py [--moveatlas <moveatlas>]
"""
import argparse
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moveatlas_stamp import default_root, stamp  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(TREE, "reports", "gene_classes")
RULE = os.path.join(OUT, "rule.txt")
RULE_HASH_FILE = os.path.join(OUT, "rule.sha256")
RULE_SHA256 = "2351647e28fd6384bfcad521db6d0c11cfff186b1b9ab5744f12895766651f43"
GENES_CSV = os.path.join(TREE, "reports", "pallidal_units", "patients_per_gene.csv")
HPO_REL = "datasets/ontology/hpo/raw"
OBO, HPOA, G2P = "hp.obo", "phenotype.hpoa.gz", "genes_to_phenotype.txt"
TITLE = "gene class or its surgical correlates"

# G6
FREQ_TERMS = {"HP:0040280": 1.0, "HP:0040281": 0.895, "HP:0040282": 0.545, "HP:0040283": 0.17,
              "HP:0040284": 0.025, "HP:0040285": 0.0}
FREQ_NAMES = {"HP:0040280": "Obligate", "HP:0040281": "Very frequent", "HP:0040282": "Frequent",
              "HP:0040283": "Occasional", "HP:0040284": "Very rare", "HP:0040285": "Excluded"}
W_BLANK = 0.5
THETA = 0.3
# G8
CATEGORIES = {
    "DYS": (("HP:0001332", "Dystonia"),),
    "MYO": (("HP:0001336", "Myoclonus"),),
    "PAR": (("HP:0001300", "Parkinsonism"),),
    "SPA": (("HP:0001257", "Spasticity"),),
    "NDG": (("HP:0002180", "Neurodegeneration"), ("HP:0002376", "Developmental regression")),
    "CHO": (("HP:0002072", "Chorea"),),
    "ATA": (("HP:0001251", "Ataxia"),),
}
COMBINING = ("MYO", "PAR", "SPA", "NDG")
COMBINING_V4 = ("MYO", "PAR", "SPA", "NDG", "CHO", "ATA")
# G12: name -> (w_blank, theta, combining categories)
VARIANTS = {
    "primary": (W_BLANK, THETA, COMBINING),
    "V1": (0.0, THETA, COMBINING),
    "V2": (1.0, THETA, COMBINING),
    "V3": (W_BLANK, 0.8, COMBINING),
    "V4": (W_BLANK, THETA, COMBINING_V4),
}
# G11
BRAIN_IRON = ("PANK2", "PLA2G6")
BRAIN_IRON_COMPARATOR = ("TOR1A", "THAP1", "GNAL")
ENUM_MAX = 10 ** 6      # reporting only: the enumeration limit of probe_oxford_structure.py (T11), not a rule item
ATTRIBUTION_URL = "https://hpo.jax.org/"
CITATION = "Gargano et al., Nucleic Acids Research 2024, doi 10.1093/nar/gkad1005"


# ----------------------------------------------------------------------------- guards

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_rule_hash(rule=RULE, hash_file=RULE_HASH_FILE, expected=RULE_SHA256):
    """Stop before any input is read when rule.txt is not the hashed version."""
    actual = sha256_file(rule)
    with open(hash_file, encoding="utf-8") as f:
        recorded = f.read().split()[0]
    assert actual == expected == recorded, f"rule.txt {actual}, rule.sha256 {recorded}, script {expected}"
    return actual


# ----------------------------------------------------------------------------- ontology

def parse_obo(lines):
    """(terms, data_version). terms: id -> {name, is_a, obsolete, replaced_by, alt_ids}."""
    terms, version, cur, in_term = {}, None, None, False
    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if line.startswith("data-version:") and version is None:
            version = line.split(":", 1)[1].strip()
        if line.startswith("["):
            in_term = line.strip() == "[Term]"
            cur = None
            continue
        if not in_term or not line or line.startswith("!"):
            continue
        tag, _, value = line.partition(":")
        value = value.strip()
        if tag == "id":
            cur = {"name": "", "is_a": [], "obsolete": False, "replaced_by": [], "alt_ids": []}
            terms[value] = cur
        elif cur is None:
            continue
        elif tag == "name":
            cur["name"] = value
        elif tag == "is_a":
            cur["is_a"].append(value.split("!")[0].split("{")[0].strip())
        elif tag == "is_obsolete":
            cur["obsolete"] = value.lower() == "true"
        elif tag == "replaced_by":
            cur["replaced_by"].append(value.split()[0])
        elif tag == "alt_id":
            cur["alt_ids"].append(value.split()[0])
    return terms, version


class Ontology:
    def __init__(self, terms):
        self.terms = terms
        self.alt = {}
        for tid, t in terms.items():
            for a in t["alt_ids"]:
                self.alt[a] = tid
        self._anc = {}

    def primary(self, tid):
        """(primary id or None, reason) per G4."""
        if tid in self.alt and tid not in self.terms:
            tid = self.alt[tid]
        t = self.terms.get(tid)
        if t is None:
            return None, "unknown"
        if t["obsolete"]:
            if len(t["replaced_by"]) == 1 and t["replaced_by"][0] in self.terms \
                    and not self.terms[t["replaced_by"][0]]["obsolete"]:
                return t["replaced_by"][0], "replaced"
            return None, "obsolete"
        return tid, "ok"

    def ancestors(self, tid):
        """tid and all its is_a ancestors (G5)."""
        if tid in self._anc:
            return self._anc[tid]
        out, stack = {tid}, [tid]
        while stack:
            t = stack.pop()
            for p in self.terms.get(t, {}).get("is_a", []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        self._anc[tid] = frozenset(out)
        return self._anc[tid]

    def check_roots(self):
        for cat, roots in CATEGORIES.items():
            for tid, name in roots:
                t = self.terms.get(tid)
                assert t is not None and not t["obsolete"], f"{cat}: {tid} missing or obsolete in hp.obo"
                assert t["name"] == name, f"{cat}: {tid} is '{t['name']}', rule says '{name}'"
        for tid, name in FREQ_NAMES.items():
            t = self.terms.get(tid)
            assert t is not None and t["name"] == name, f"frequency term {tid} is not '{name}'"


# ----------------------------------------------------------------------------- annotations

_FRAC = re.compile(r"^(\d+)\s*/\s*(\d+)$")
_PCT = re.compile(r"^(\d+(?:\.\d+)?)\s*%$")


def weight(freq, w_blank=W_BLANK):
    """Frequency weight of an annotation row (G6); ValueError on any other string."""
    f = (freq or "").strip()
    if f == "":
        return w_blank
    if f in FREQ_TERMS:
        return FREQ_TERMS[f]
    m = _FRAC.match(f)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        if d <= 0 or n > d:
            raise ValueError(f"frequency {f!r}")
        return n / d
    m = _PCT.match(f)
    if m:
        x = float(m.group(1))
        if x > 100:
            raise ValueError(f"frequency {f!r}")
        return x / 100.0
    raise ValueError(f"frequency {f!r} is outside rule G6")


def read_gene_entities(lines, genes):
    """gene -> {"ncbi": set, "entities": set} from genes_to_phenotype.txt rows (G1, G3)."""
    reader = csv.DictReader(lines, delimiter="\t")
    need = {"ncbi_gene_id", "gene_symbol", "disease_id"}
    assert need <= set(reader.fieldnames or []), reader.fieldnames
    out = {g: {"ncbi": set(), "entities": set()} for g in genes}
    for r in reader:
        g = r["gene_symbol"]
        if g in out:
            out[g]["ncbi"].add(r["ncbi_gene_id"])
            if r["disease_id"]:
                out[g]["entities"].add(r["disease_id"])
    return out


def read_hpoa(lines, entities):
    """(version lines, entity -> {"name", "rows": [(hpo_id, frequency, qualifier, aspect)]})."""
    meta, body = {}, []
    for line in lines:
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            meta[k.strip()] = v.strip()
        else:
            body.append(line)
    reader = csv.DictReader(body, delimiter="\t")
    need = {"database_id", "disease_name", "qualifier", "hpo_id", "frequency", "aspect"}
    assert need <= set(reader.fieldnames or []), reader.fieldnames
    out = {e: {"name": "", "rows": []} for e in entities}
    for r in reader:
        e = r["database_id"]
        if e in out:
            out[e]["name"] = out[e]["name"] or r["disease_name"]
            out[e]["rows"].append((r["hpo_id"], r["frequency"], r["qualifier"], r["aspect"]))
    return meta, out


def entity_terms(onto, rows):
    """(kept [(primary id, frequency string)], counts) per G4."""
    kept, counts = [], {"rows": len(rows), "not": 0, "other_aspect": 0, "replaced": 0, "obsolete_dropped": 0,
                        "unknown_dropped": 0, "blank_frequency": 0}
    for hpo_id, freq, qual, aspect in rows:
        if aspect != "P":
            counts["other_aspect"] += 1
            continue
        if qual.strip():
            assert qual.strip() == "NOT", f"qualifier {qual!r}"
            counts["not"] += 1
            continue
        tid, why = onto.primary(hpo_id)
        if tid is None:
            counts["obsolete_dropped" if why == "obsolete" else "unknown_dropped"] += 1
            continue
        if why == "replaced":
            counts["replaced"] += 1
        if not freq.strip():
            counts["blank_frequency"] += 1
        kept.append((tid, freq.strip()))
    counts["kept"] = len(kept)
    return kept, counts


def inheritance(onto, rows):
    """Sorted names of the aspect I (mode of inheritance) terms of an entity, for reading only: no rule item
    uses them and they take no part in any class."""
    names = set()
    for hpo_id, _freq, qual, aspect in rows:
        if aspect != "I" or qual.strip():
            continue
        tid, _ = onto.primary(hpo_id)
        names.add(onto.terms[tid]["name"] if tid else hpo_id)
    return sorted(names)


def category_scores(onto, kept, w_blank):
    """cat -> (score, evidence list of 'id name [frequency]', number of kept rows under the roots) per G7.
    The row count is reported for reading only and takes no part in the class."""
    out = {}
    for cat, roots in CATEGORIES.items():
        rootset = {r for r, _ in roots}
        best, ev, n = 0.0, [], 0
        for tid, freq in kept:
            if not (onto.ancestors(tid) & rootset):
                continue
            n += 1
            w = weight(freq, w_blank)
            label = f"{tid} {onto.terms[tid]['name']} [{freq or 'blank'}]"
            if w > best:
                best, ev = w, [label]
            elif w == best and w > 0 and label not in ev:
                ev.append(label)
        out[cat] = (best, sorted(ev), n)
    return out


def entity_class(scores, theta, combining):
    """G9: not_dystonia_bearing, combined or isolated."""
    if scores["DYS"][0] < theta:
        return "not_dystonia_bearing"
    if any(scores[c][0] >= theta for c in combining):
        return "combined"
    return "isolated"


def gene_class(entity_classes, has_gene):
    """G10."""
    if not has_gene:
        return "no_hpo_gene"
    dys = [c for c in entity_classes if c != "not_dystonia_bearing"]
    if not dys:
        return "no_dystonia_entity"
    if all(c == "isolated" for c in dys):
        return "isolated"
    if all(c == "combined" for c in dys):
        return "combined"
    return "mixed"


def contrast_count(n_a, n_b):
    """G13: k = C(n_a + n_b, n_b) - 1 and the smallest p 1 / (k + 1)."""
    if n_a <= 0 or n_b <= 0:
        return 0, None
    k = math.comb(n_a + n_b, n_b) - 1
    return k, Fraction(1, k + 1)


# ----------------------------------------------------------------------------- build

def build(onto, genes_rows, g2e, hpoa):
    """(entity rows, class rows)."""
    ent_rows, cls_rows = [], []
    for g in genes_rows:
        gene = g["gene"]
        info = g2e.get(gene, {"ncbi": set(), "entities": set()})
        has_gene = bool(info["ncbi"])
        classes = {v: [] for v in VARIANTS}
        dys_list = []
        for e in sorted(info["entities"]):
            rec = hpoa.get(e, {"name": "", "rows": []})
            kept, counts = entity_terms(onto, rec["rows"])
            by_blank = {wb: category_scores(onto, kept, wb) for wb in sorted({v[0] for v in VARIANTS.values()})}
            prim = by_blank[W_BLANK]
            row = {"gene": gene, "disease_id": e, "disease_name": rec["name"], "hpoa_rows": counts["rows"],
                   "rows_kept": counts["kept"], "rows_not": counts["not"], "rows_other_aspect": counts["other_aspect"],
                   "rows_replaced": counts["replaced"], "rows_obsolete_dropped": counts["obsolete_dropped"],
                   "rows_unknown_dropped": counts["unknown_dropped"], "rows_blank_frequency": counts["blank_frequency"]}
            for cat in CATEGORIES:
                row[f"s_{cat}"] = f"{prim[cat][0]:.4g}"
            for cat in CATEGORIES:
                row[f"n_rows_{cat}"] = prim[cat][2]
            for v, (wb, th, comb) in VARIANTS.items():
                c = entity_class(by_blank[wb], th, comb)
                classes[v].append(c)
                row["entity_class" if v == "primary" else f"entity_class_{v}"] = c
            for cat in CATEGORIES:
                row[f"evidence_{cat}"] = "; ".join(prim[cat][1])
            row["inheritance"] = "; ".join(inheritance(onto, rec["rows"]))
            if row["entity_class"] != "not_dystonia_bearing":
                dys_list.append(f"{e}:{row['entity_class']}")
            ent_rows.append(row)
        gc = {v: gene_class(classes[v], has_gene) for v in VARIANTS}
        unstable = [v for v in VARIANTS if v != "primary" and gc[v] != gc["primary"]]
        cls_rows.append({
            "gene": gene, "ncbi_gene_id": " ".join(sorted(info["ncbi"])), "n_patients": int(g["n_patients"]),
            "n_units": int(g["n_units"]), "n_entities": len(info["entities"]),
            "n_dystonia_entities": len(dys_list), "dystonia_entities": " ".join(dys_list),
            "class": gc["primary"], "class_V1": gc["V1"], "class_V2": gc["V2"], "class_V3": gc["V3"],
            "class_V4": gc["V4"], "stable": "yes" if not unstable else "no (" + " ".join(unstable) + ")",
            "brain_iron": "yes" if gene in BRAIN_IRON else "no",
            "contrast_b_group": ("brain_iron" if gene in BRAIN_IRON else
                                 "comparator" if gene in BRAIN_IRON_COMPARATOR else "none"),
        })
    return ent_rows, cls_rows


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x):
    if isinstance(x, Fraction):
        x = float(x)
    return f"{x:.6g}"


def stamp_line(s):
    """The MoveAtlas commit line of a summary, from a stamp of scripts/moveatlas_stamp.py (no time field)."""
    return (f"- MoveAtlas measured against: HEAD {s['head']}, uncommitted diff SHA-256 {s['diff_sha256']} "
            f"({s.get('diff_bytes', '?')} bytes), from scripts/moveatlas_stamp.py; the full stamp is in "
            "_provenance.json.")


ROW_COUNTS = ("hpoa_rows", "rows_kept", "rows_not", "rows_other_aspect", "rows_obsolete_dropped",
              "rows_unknown_dropped", "rows_replaced", "rows_blank_frequency")


def pair_counts(ent_rows):
    """Annotation row counts summed over the gene-entity pairs and over the distinct entities, and the entities
    linked to more than one gene (entity -> genes)."""
    first, genes = {}, {}
    for e in ent_rows:
        first.setdefault(e["disease_id"], e)
        genes.setdefault(e["disease_id"], []).append(e["gene"])
    by_pair = {c: sum(int(e[c]) for e in ent_rows) for c in ROW_COUNTS}
    by_entity = {c: sum(int(e[c]) for e in first.values()) for c in ROW_COUNTS}
    shared = {d: g for d, g in sorted(genes.items()) if len(g) > 1}
    return by_pair, by_entity, len(first), shared


def summary_md(cls_rows, ent_rows, versions, in_hashes, out_hashes, rule_sha, ma_stamp):
    L = []
    L.append(f"# {TITLE[0].upper() + TITLE[1:]}: HPO classes of the pallidal_units_dystonia genes")
    L.append("")
    L.append("Status: DRAFT, pending the neurologist (design note 2026-09-29, priority 7 and question 18). The "
             "classes are fixed by script before any firing value is read; they are literature-level classes of "
             "disease entities, not the phenotype of the cohort's patients.")
    L.append("")
    L.append(f"- Rule: reports/gene_classes/rule.txt, SHA-256 {rule_sha} (rule.sha256), checked before any input "
             "was read. Script: scripts/build_gene_classes.py.")
    L.append(stamp_line(ma_stamp))
    L.append(f"- Primary rule: frequency weights of G6 (blank {W_BLANK}), largest weight per category over "
             f"propagated is_a descendants (G5, G7), threshold {THETA} (G9); combining categories "
             + ", ".join(COMBINING) + "; gene class from its dystonia-bearing entities (G10).")
    L.append("")
    L.append("## HPO versions and attribution")
    L.append("")
    L.append(f"- hp.obo: data-version {versions['obo']}; SHA-256 {in_hashes[OBO]}.")
    L.append(f"- phenotype.hpoa.gz: version {versions['hpoa']}, hpo-version {versions['hpoa_hpo']}; SHA-256 "
             f"{in_hashes[HPOA]}.")
    L.append(f"- genes_to_phenotype.txt: no version line in the file; SHA-256 {in_hashes[G2P]}.")
    rel = versions["obo"].split("/")[-1] if versions["obo"] else "unknown"
    L.append(f"- Attribution: this report uses the Human Phenotype Ontology (release {rel}). Find out more at "
             f"{ATTRIBUTION_URL}. Cite {CITATION}. The HPO files were read unmodified.")
    L.append("")
    L.append("## Gene-entity pairs and annotation rows")
    L.append("")
    by_pair, by_ent, n_distinct, shared = pair_counts(ent_rows)
    n_genes = len({e["gene"] for e in ent_rows})
    L.append(f"- {len(ent_rows)} gene-entity pairs over {n_genes} genes, {n_distinct} distinct entities"
             + ("; linked to more than one gene: " + "; ".join(f"{d} ({', '.join(g)})" for d, g in shared.items())
                if shared else "; no entity is linked to more than one gene") + " (entities.csv).")
    for label, c in (("summed over the gene-entity pairs", by_pair), ("over the distinct entities", by_ent)):
        L.append(f"- Annotation rows {label}: {c['hpoa_rows']}; kept {c['rows_kept']} (of which "
                 f"{c['rows_blank_frequency']} with a blank frequency and {c['rows_replaced']} mapped from an "
                 f"obsolete term); dropped: {c['rows_not']} NOT, {c['rows_other_aspect']} of an aspect other than P, "
                 f"{c['rows_obsolete_dropped']} obsolete, {c['rows_unknown_dropped']} unknown.")
    L.append("")
    L.append("## Classes")
    L.append("")
    L.append("| gene | patients | units | entities | dystonia-bearing | class | V1 blank 0 | V2 blank 1 | "
             "V3 theta 0.8 | V4 + chorea, ataxia | stable | brain iron |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cls_rows:
        L.append(f"| {r['gene']} | {r['n_patients']} | {r['n_units']} | {r['n_entities']} | "
                 f"{r['n_dystonia_entities']} | {r['class']} | {r['class_V1']} | {r['class_V2']} | {r['class_V3']} | "
                 f"{r['class_V4']} | {r['stable']} | {r['brain_iron']} |")
    L.append("")
    L.append("## Dystonia-bearing entities (primary rule)")
    L.append("")
    for r in cls_rows:
        ents = [e for e in ent_rows if e["gene"] == r["gene"] and e["entity_class"] != "not_dystonia_bearing"]
        if not ents:
            L.append(f"- {r['gene']}: none.")
            continue
        for e in ents:
            comb = [f"{c} {e['s_' + c]} ({e['evidence_' + c]})" for c in COMBINING if float(e["s_" + c]) >= THETA]
            L.append(f"- {r['gene']} {e['disease_id']} {e['disease_name']}: {e['entity_class']}; dystonia "
                     f"{e['s_DYS']}" + (f"; {'; '.join(comb)}" if comb else "") + ".")
    others = sum(1 for e in ent_rows if e["entity_class"] == "not_dystonia_bearing")
    L.append(f"- Gene-entity pairs not dystonia-bearing under the primary rule: {others} of {len(ent_rows)} "
             "(entities.csv).")
    single = [e for e in ent_rows if e["entity_class"] != "not_dystonia_bearing" and int(e["n_rows_DYS"]) == 1]
    if single:
        L.append("- For the neurologist, dystonia-bearing pairs whose dystonia score rests on one annotation row "
                 "(n_rows_DYS of entities.csv; inheritance from the aspect I rows, read only): " + "; ".join(
                     f"{e['gene']} {e['disease_id']} {e['disease_name']}, {e['evidence_DYS']}, class "
                     f"{e['entity_class']}, inheritance {e['inheritance'] or 'not annotated'}" for e in single)
                 + ".")
    L.append("")
    L.append(f"## Patients per class and contrast counts ({TITLE})")
    L.append("")
    by = {}
    for r in cls_rows:
        by.setdefault(r["class"], []).append(r)
    for c in sorted(by):
        L.append(f"- {c}: {sum(r['n_patients'] for r in by[c])} patients, "
                 f"{sum(r['n_units'] for r in by[c])} units ("
                 + ", ".join(f"{r['gene']} {r['n_patients']}" for r in by[c]) + ").")
    n_iso = sum(r["n_patients"] for r in by.get("isolated", []))
    n_comb = sum(r["n_patients"] for r in by.get("combined", []))
    k, p = contrast_count(n_iso, n_comb)
    if k:
        enum = (f"above {ENUM_MAX}, the enumeration limit of oxford_q1.csv: with B drawn nulls the smallest p is "
                f"1 / (B + 1)" if k > ENUM_MAX else f"at most {ENUM_MAX}, enumerable")
        L.append(f"- Contrast (a), isolated against combined: {n_iso} and {n_comb} patients, "
                 f"k = C({n_iso} + {n_comb}, {n_comb}) - 1 = {k} ({enum}), smallest p under full enumeration "
                 f"{fmt(p)} (G13; exploratory, outside the Holm family).")
    else:
        L.append(f"- Contrast (a), isolated against combined: {n_iso} and {n_comb} patients; no contrast.")
    out_a = [r for r in cls_rows if r["class"] not in ("isolated", "combined")]
    if out_a:
        L.append("- Outside contrast (a) (G13): " + ", ".join(
            f"{r['gene']} {r['class']} ({r['n_patients']} patients)" for r in out_a) + ".")
    n_bi = sum(r["n_patients"] for r in cls_rows if r["contrast_b_group"] == "brain_iron")
    n_cmp = sum(r["n_patients"] for r in cls_rows if r["contrast_b_group"] == "comparator")
    k, p = contrast_count(n_bi, n_cmp)
    L.append(f"- Contrast (b), brain iron ({', '.join(BRAIN_IRON)}, explicit list) against "
             f"{', '.join(BRAIN_IRON_COMPARATOR)}: {n_bi} and {n_cmp} patients, k = {k}, smallest p "
             f"{fmt(p) if p else 'none'} (G11, G13; exploratory).")
    L.append("")
    L.append("## Output SHA-256")
    L.append("")
    for k2, v in out_hashes.items():
        L.append(f"- {k2}: {v}")
    L.append("- summary.md: in _provenance.json.")
    L.append("")
    L.append("## Limits")
    L.append("")
    L.append("- The class is literature level (curated annotations of disease entities), not the phenotype of the "
             "cohort's patients. Gene is nested in class; age at surgery, anaesthesia, centre and era covary with "
             "gene and are absent from the file, so every contrast is titled \"" + TITLE + "\".")
    L.append("- The weights, the threshold and the categories are the rule's choices; the variant columns show "
             "which genes change class when one of them changes. Annotation density still tracks publication "
             "volume.")
    L.append("")
    return "\n".join(L)


def run(moveatlas):
    rule_sha = assert_rule_hash()
    hpo = os.path.join(moveatlas, HPO_REL)
    genes_rows = []
    with open(GENES_CSV, encoding="utf-8", newline="") as f:
        genes_rows = list(csv.DictReader(f))
    genes = [r["gene"] for r in genes_rows]
    in_hashes = {n: sha256_file(os.path.join(hpo, n)) for n in (OBO, HPOA, G2P)}
    with open(os.path.join(hpo, OBO), encoding="utf-8") as f:
        terms, obo_version = parse_obo(f)
    onto = Ontology(terms)
    onto.check_roots()
    with open(os.path.join(hpo, G2P), encoding="utf-8", newline="") as f:
        g2e = read_gene_entities(f, genes)
    entities = set()
    for v in g2e.values():
        entities |= v["entities"]
    with gzip.open(os.path.join(hpo, HPOA), "rb") as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        meta, hpoa = read_hpoa(text, entities)
    ent_rows, cls_rows = build(onto, genes_rows, g2e, hpoa)
    os.makedirs(OUT, exist_ok=True)
    out_hashes = {}
    for name, rows in (("entities.csv", ent_rows), ("classes.csv", cls_rows)):
        p = os.path.join(OUT, name)
        write_csv(p, rows)
        out_hashes[name] = sha256_file(p)
    versions = {"obo": obo_version, "hpoa": meta.get("version", "absent"), "hpoa_hpo": meta.get("hpo-version", "absent")}
    ma_stamp = stamp(moveatlas)
    text = summary_md(cls_rows, ent_rows, versions, in_hashes, out_hashes, rule_sha, ma_stamp)
    assert "\u2014" not in text and "\u2013" not in text
    p = os.path.join(OUT, "summary.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out_hashes["summary.md"] = sha256_file(p)
    prov = {
        "script": "scripts/build_gene_classes.py",
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "rule": "reports/gene_classes/rule.txt",
        "rule_sha256": rule_sha,
        "moveatlas_stamp": ma_stamp,
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "hpo_versions": versions,
        "inputs": {**{f"{HPO_REL}/{k}": v for k, v in in_hashes.items()},
                   "reports/pallidal_units/patients_per_gene.csv": sha256_file(GENES_CSV)},
        "parameters": {"W_BLANK": W_BLANK, "THETA": THETA, "FREQ_TERMS": FREQ_TERMS,
                       "CATEGORIES": {k: [t for t, _ in v] for k, v in CATEGORIES.items()},
                       "COMBINING": list(COMBINING), "VARIANTS": {k: [v[0], v[1], list(v[2])] for k, v in VARIANTS.items()},
                       "BRAIN_IRON": list(BRAIN_IRON), "BRAIN_IRON_COMPARATOR": list(BRAIN_IRON_COMPARATOR)},
        "outputs": {f"reports/gene_classes/{k}": v for k, v in out_hashes.items()},
        "note": "status DRAFT pending the neurologist; no row of the pallidal CSV read (G15)",
    }
    with open(os.path.join(OUT, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    return cls_rows, ent_rows, out_hashes, rule_sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--moveatlas", default=default_root())
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cls_rows, ent_rows, outs, rule_sha = run(a.moveatlas.replace("\\", "/").rstrip("/"))
    print(f"rule {rule_sha}")
    for r in cls_rows:
        print(f"  {r['gene']}: {r['class']} ({r['n_dystonia_entities']} of {r['n_entities']} entities "
              f"dystonia-bearing; stable {r['stable']})")
    for k, v in outs.items():
        print(f"  {k} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
