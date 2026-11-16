"""Tests of scripts/build_gene_classes.py (task T12) on a synthetic ontology and synthetic annotation files;
no MoveAtlas file is read.

They check the rule-hash guard (the constant equals rule.txt and rule.sha256, a changed rule stops the run,
the guard runs before any input is read), that every script constant that decides a class equals the value
written in the hashed rule.txt (G6, G8, G9, G11, G12), the obo parse, alt_id and obsolete handling (G4),
propagation (G5), the frequency weights (G6), the category score and the entity and gene classes (G7 to
G10) with their boundaries at THETA, the variants (G12), the brain-iron columns (G11), the contrast counts
(G13) and the contrast lines of summary.md.
"""
import hashlib
import importlib.util
import inspect
import math
import os
import re
from fractions import Fraction

import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "build_gene_classes.py")


def _script():
    spec = importlib.util.spec_from_file_location("build_gene_classes_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


def _term(tid, name, parents=(), extra=()):
    lines = ["[Term]", f"id: {tid}", f"name: {name}"]
    lines += [f"is_a: {p} ! parent" for p in parents]
    lines += list(extra)
    return lines + [""]


def _obo():
    lines = ["format-version: 1.2", "data-version: hp/releases/2099-01-01", ""]
    lines += _term("HP:0000001", "All")
    lines += _term("HP:0000118", "Phenotypic abnormality", ["HP:0000001"])
    for roots in S.CATEGORIES.values():
        for tid, name in roots:
            lines += _term(tid, name, ["HP:0000118"])
    lines += _term("HP:0040279", "Frequency", ["HP:0000001"])
    for tid, name in S.FREQ_NAMES.items():
        lines += _term(tid, name, ["HP:0040279"])
    lines += _term("HP:0000005", "Mode of inheritance", ["HP:0000001"])
    lines += _term("HP:0000007", "Autosomal recessive inheritance", ["HP:0000005"])
    lines += _term("HP:0002451", "Limb dystonia", ["HP:0001332"], ["alt_id: HP:9999991"])
    lines += _term("HP:0010531", "Spinal myoclonus", ["HP:0001336"])
    lines += _term("HP:9999992", "obsolete limb dystonia", [], ["is_obsolete: true", "replaced_by: HP:0002451"])
    lines += _term("HP:9999993", "obsolete thing", [], ["is_obsolete: true"])
    lines += ["[Typedef]", "id: part_of", "name: part of", ""]
    return [line + "\n" for line in lines]


HPOA_HEADER = ("database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\t"
               "aspect\tbiocuration")


def _hpoa_lines():
    rows = [
        # E1: isolated (myoclonus only Occasional)
        ("OMIM:1", "E one", "", "HP:0002451", "HP:0040281", "P"),
        ("OMIM:1", "E one", "", "HP:0001336", "HP:0040283", "P"),
        # E2: combined under the primary rule through a blank-frequency myoclonus child
        ("OMIM:2", "E two", "", "HP:0001332", "3/3", "P"),
        ("OMIM:2", "E two", "", "HP:0010531", "", "P"),
        ("OMIM:2", "E two", "", "HP:0000007", "", "I"),
        # E3: not dystonia-bearing
        ("ORPHA:3", "E three", "", "HP:0001332", "HP:0040283", "P"),
        # E4: dystonia through an alt_id and an obsolete replaced term; a NOT myoclonus row is dropped
        ("OMIM:4", "E four", "", "HP:9999991", "1/1", "P"),
        ("OMIM:4", "E four", "", "HP:9999992", "50%", "P"),
        ("OMIM:4", "E four", "NOT", "HP:0001336", "", "P"),
        ("OMIM:4", "E four", "", "HP:9999993", "", "P"),
        ("OMIM:4", "E four", "", "HP:0001336", "", "I"),
    ]
    out = ["#description: test\n", "#version: 2099-01-01\n", "#hpo-version: http://x/2099-01-01/hp.json\n",
           HPOA_HEADER + "\n"]
    for db, name, qual, hpo, freq, aspect in rows:
        out.append("\t".join([db, name, qual, hpo, "PMID:1", "PCS", "", freq, "", "", aspect, "x"]) + "\n")
    return out


def _g2p_lines():
    out = ["ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id\n"]
    for gid, g, d in (("1", "GA", "OMIM:1"), ("1", "GA", "ORPHA:3"), ("2", "GB", "OMIM:1"), ("2", "GB", "OMIM:2"),
                      ("3", "GC", "OMIM:2"), ("4", "PANK2", "OMIM:4"), ("9", "OTHER", "OMIM:2")):
        out.append("\t".join([gid, g, "HP:0001332", "Dystonia", "-", d]) + "\n")
    return out


@pytest.fixture(scope="module")
def onto():
    terms, version = S.parse_obo(_obo())
    assert version == "hp/releases/2099-01-01"
    o = S.Ontology(terms)
    o.check_roots()
    return o


# ----------------------------------------------------------------------------- guards

def test_rule_hash_constant_matches_files():
    with open(S.RULE, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert actual == S.RULE_SHA256
    assert S.assert_rule_hash() == S.RULE_SHA256


def test_changed_rule_stops(tmp_path):
    rule = tmp_path / "rule.txt"
    rule.write_bytes(b"G1 changed\n")
    hfile = tmp_path / "rule.sha256"
    hfile.write_text(hashlib.sha256(b"G1 changed\n").hexdigest() + "  rule.txt\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        S.assert_rule_hash(str(rule), str(hfile), S.RULE_SHA256)
    assert S.assert_rule_hash(str(rule), str(hfile), hashlib.sha256(b"G1 changed\n").hexdigest())


def test_guard_runs_before_any_input():
    src = inspect.getsource(S.run)
    guard = src.index("assert_rule_hash()")
    for read in ("GENES_CSV", "sha256_file(os.path.join(hpo", "parse_obo", "gzip.open"):
        assert guard < src.index(read), read


def test_rule_text_is_ascii_and_has_no_dash():
    with open(S.RULE, "rb") as f:
        b = f.read()
    assert all(c < 128 for c in b) and b"\r" not in b


def _rule_items():
    """Item -> text (continuation lines joined) of the hashed rule.txt."""
    items, cur = {}, None
    with open(S.RULE, encoding="utf-8") as f:
        for line in f.read().splitlines():
            m = re.match(r"^(G\d+) (.*)$", line)
            if m:
                cur = m.group(1)
                items[cur] = m.group(2)
            elif cur and line.startswith("   "):
                items[cur] += " " + line.strip()
            elif not line.strip():
                cur = None
    assert sorted(items, key=lambda g: int(g[1:])) == [f"G{i}" for i in range(1, 17)]
    return items


def _gene_list(text):
    return tuple(g.strip() for g in text.split(","))


def _cat_list(head, last):
    return tuple(c.strip() for c in head.split(",")) + (last,)


def test_constants_equal_the_hashed_rule():
    """The hash guard protects rule.txt only: every constant that decides a class must equal the rule text."""
    it = _rule_items()
    # G6: frequency terms, names and weights; the blank weight.
    g6 = re.findall(r"(HP:\d{7}) ([A-Z][a-z]+(?: [a-z]+)?) (\d+(?:\.\d+)?)", it["G6"])
    assert {t: float(w) for t, _, w in g6} == S.FREQ_TERMS
    assert {t: n for t, n, _ in g6} == S.FREQ_NAMES
    assert float(re.search(r"W_BLANK = (\d+(?:\.\d+)?)", it["G6"]).group(1)) == S.W_BLANK
    # G8: category roots (ids and names), including the V4-only categories.
    cats = {}
    for cat, roots in re.findall(r"\b([A-Z]{3}) [a-z]+: (.+?)\.(?= |$)", it["G8"]):
        cats[cat] = tuple(tuple(r.split(" ", 1)) for r in roots.split("; "))
    for cat, tid, name in re.findall(r"\b([A-Z]{3}) (HP:\d{7}) ([A-Z][a-z]+)", it["G8"]):
        cats[cat] = ((tid, name),)
    assert cats == S.CATEGORIES
    # G9: threshold and combining categories.
    theta = float(re.search(r"THETA = (\d+(?:\.\d+)?)", it["G9"]).group(1))
    assert theta == S.THETA
    m = re.search(r"at least one of ([A-Z, ]+?) and ([A-Z]{3}),", it["G9"])
    assert _cat_list(*m.groups()) == S.COMBINING
    # G11: the explicit brain-iron list and its comparator.
    assert _gene_list(re.search(r"not derived from HPO: ([A-Z0-9, ]+?)\.", it["G11"]).group(1)) == S.BRAIN_IRON
    assert _gene_list(re.search(r"priority 7: ([A-Z0-9, ]+?)\.", it["G11"]).group(1)) == S.BRAIN_IRON_COMPARATOR
    # G12: the four variants.
    g12 = it["G12"]
    v1 = float(re.search(r"V1 W_BLANK = (\d+(?:\.\d+)?)", g12).group(1))
    v2 = float(re.search(r"V2 W_BLANK = (\d+(?:\.\d+)?)", g12).group(1))
    v3 = float(re.search(r"V3 THETA = (\d+(?:\.\d+)?)", g12).group(1))
    v4 = _cat_list(*re.search(r"V4 combining categories ([A-Z, ]+?) and ([A-Z]{3})\.", g12).groups())
    assert v4 == S.COMBINING_V4
    assert S.VARIANTS == {"primary": (S.W_BLANK, theta, S.COMBINING), "V1": (v1, theta, S.COMBINING),
                          "V2": (v2, theta, S.COMBINING), "V3": (S.W_BLANK, v3, S.COMBINING),
                          "V4": (S.W_BLANK, theta, v4)}
    assert (v1, v2, v3) == (0.0, 1.0, 0.8)


# ----------------------------------------------------------------------------- ontology and weights

def test_obo_alt_obsolete_and_ancestors(onto):
    assert onto.primary("HP:9999991") == ("HP:0002451", "ok")
    assert onto.primary("HP:9999992") == ("HP:0002451", "replaced")
    assert onto.primary("HP:9999993") == (None, "obsolete")
    assert onto.primary("HP:1234567") == (None, "unknown")
    assert "HP:0001332" in onto.ancestors("HP:0002451")
    assert "HP:0001336" not in onto.ancestors("HP:0002451")
    assert "part_of" not in onto.terms


def test_check_roots_stops_on_a_wrong_name():
    terms, _ = S.parse_obo(_obo())
    terms["HP:0001336"]["name"] = "Something else"
    with pytest.raises(AssertionError):
        S.Ontology(terms).check_roots()


def test_weights():
    assert S.weight("HP:0040280") == 1.0 and S.weight("HP:0040282") == 0.545
    assert S.weight("HP:0040285") == 0.0
    assert S.weight("3/4") == 0.75 and S.weight("25%") == 0.25
    assert S.weight("") == S.W_BLANK and S.weight("", 0.0) == 0.0 and S.weight("", 1.0) == 1.0
    for bad in ("often", "5/0", "4/3", "120%"):
        with pytest.raises(ValueError):
            S.weight(bad)


# ----------------------------------------------------------------------------- classes

def test_entity_and_gene_classes(onto):
    g2e = S.read_gene_entities(_g2p_lines(), ["GA", "GB", "GC", "PANK2", "NOHPO"])
    assert g2e["GA"]["entities"] == {"OMIM:1", "ORPHA:3"} and g2e["NOHPO"]["ncbi"] == set()
    entities = set().union(*(v["entities"] for v in g2e.values()))
    meta, hpoa = S.read_hpoa(_hpoa_lines(), entities)
    assert meta["version"] == "2099-01-01"
    genes_rows = [{"gene": g, "n_patients": str(n), "n_units": str(10 * n)}
                  for g, n in (("GA", 4), ("GB", 3), ("GC", 2), ("PANK2", 1), ("NOHPO", 5))]
    ent_rows, cls_rows = S.build(onto, genes_rows, g2e, hpoa)
    ent = {(r["gene"], r["disease_id"]): r for r in ent_rows}
    assert ent[("GA", "OMIM:1")]["entity_class"] == "isolated"
    assert ent[("GA", "OMIM:1")]["s_MYO"] == "0.17"
    assert ent[("GA", "ORPHA:3")]["entity_class"] == "not_dystonia_bearing"
    assert ent[("GB", "OMIM:2")]["entity_class"] == "combined"
    assert ent[("GB", "OMIM:2")]["entity_class_V1"] == "isolated"
    e4 = ent[("PANK2", "OMIM:4")]
    assert e4["entity_class"] == "isolated" and e4["s_DYS"] == "1"
    assert e4["rows_not"] == 1 and e4["rows_replaced"] == 1 and e4["rows_obsolete_dropped"] == 1
    assert e4["rows_other_aspect"] == 1
    # reading-only columns: kept rows under each category, inheritance from the aspect I rows
    assert e4["n_rows_DYS"] == 2 and e4["n_rows_MYO"] == 0
    assert ent[("GA", "OMIM:1")]["n_rows_DYS"] == 1 and ent[("GA", "OMIM:1")]["n_rows_MYO"] == 1
    assert ent[("GB", "OMIM:2")]["inheritance"] == "Autosomal recessive inheritance"
    assert ent[("GB", "OMIM:2")]["rows_other_aspect"] == 1 and ent[("GA", "OMIM:1")]["inheritance"] == ""
    assert e4["inheritance"] == "Myoclonus"        # an aspect I row is named, whatever its term
    cls = {r["gene"]: r for r in cls_rows}
    assert cls["GA"]["class"] == "isolated"
    assert cls["GB"]["class"] == "mixed"
    assert cls["GC"]["class"] == "combined" and cls["GC"]["class_V1"] == "isolated"
    assert cls["GC"]["stable"].startswith("no") and "V1" in cls["GC"]["stable"]
    assert cls["NOHPO"]["class"] == "no_hpo_gene"
    assert cls["PANK2"]["brain_iron"] == "yes" and cls["PANK2"]["contrast_b_group"] == "brain_iron"
    assert cls["GA"]["contrast_b_group"] == "none"


def test_entity_class_boundaries_at_theta():
    """G9 says >= THETA for dystonia and for a combining category."""
    zero = {c: (0.0, [], 0) for c in S.CATEGORIES}
    at = dict(zero, DYS=(S.THETA, [], 1))
    below = math.nextafter(S.THETA, 0.0)
    assert S.entity_class(at, S.THETA, S.COMBINING) == "isolated"
    assert S.entity_class(dict(zero, DYS=(below, [], 1)), S.THETA, S.COMBINING) == "not_dystonia_bearing"
    for c in S.COMBINING:
        assert S.entity_class(dict(at, **{c: (S.THETA, [], 1)}), S.THETA, S.COMBINING) == "combined", c
        assert S.entity_class(dict(at, **{c: (below, [], 1)}), S.THETA, S.COMBINING) == "isolated", c
    for c in ("CHO", "ATA"):
        assert S.entity_class(dict(at, **{c: (1.0, [], 1)}), S.THETA, S.COMBINING) == "isolated", c
        assert S.entity_class(dict(at, **{c: (1.0, [], 1)}), S.THETA, S.COMBINING_V4) == "combined", c
    # weights that land on THETA exactly
    assert S.weight("3/10") == S.THETA and S.weight("30%") == S.THETA


def test_gene_class_rule():
    assert S.gene_class([], False) == "no_hpo_gene"
    assert S.gene_class(["not_dystonia_bearing"], True) == "no_dystonia_entity"
    assert S.gene_class(["isolated", "not_dystonia_bearing"], True) == "isolated"
    assert S.gene_class(["combined", "combined"], True) == "combined"
    assert S.gene_class(["combined", "isolated"], True) == "mixed"


def test_contrast_count():
    k, p = S.contrast_count(13, 10)
    assert k == 1144065
    assert p == Fraction(1, k + 1)
    assert S.contrast_count(0, 5) == (0, None)


STAMP = {"root": "x", "head": "h" * 40, "diff_sha256": "d" * 64, "diff_bytes": 5, "time": "2099-01-01T00:00:00"}


def test_summary_renders(onto):
    genes = ("GA", "GB", "GC", "PANK2", "THAP1")
    g2e = S.read_gene_entities(_g2p_lines(), list(genes))
    entities = set().union(*(v["entities"] for v in g2e.values()))
    _, hpoa = S.read_hpoa(_hpoa_lines(), entities)
    # GA isolated 4, GB mixed 3, GC combined 2, PANK2 isolated and brain iron 1, THAP1 no_hpo_gene and comparator 3
    genes_rows = [{"gene": g, "n_patients": str(n), "n_units": str(10 * n)}
                  for g, n in zip(genes, (4, 3, 2, 1, 3))]
    ent_rows, cls_rows = S.build(onto, genes_rows, g2e, hpoa)
    text = S.summary_md(cls_rows, ent_rows, {"obo": "hp/releases/2099-01-01", "hpoa": "2099-01-01",
                                             "hpoa_hpo": "x"},
                        {S.OBO: "a", S.HPOA: "b", S.G2P: "c"}, {"entities.csv": "d", "classes.csv": "e"}, "f",
                        STAMP)
    assert text.startswith("# Gene class or its surgical correlates")
    assert "Status: DRAFT, pending the neurologist" in text
    assert "release 2099-01-01" in text and "https://hpo.jax.org/" in text
    assert f"MoveAtlas measured against: HEAD {'h' * 40}, uncommitted diff SHA-256 {'d' * 64} (5 bytes)" in text
    assert "2099-01-01T00:00:00" not in text
    # pairs, distinct entities and annotation rows (OMIM:1 2 rows, OMIM:2 3, ORPHA:3 1, OMIM:4 5)
    assert ("- 6 gene-entity pairs over 4 genes, 4 distinct entities; linked to more than one gene: "
            "OMIM:1 (GA, GB); OMIM:2 (GB, GC) (entities.csv).") in text
    assert "- Annotation rows summed over the gene-entity pairs: 16; kept 11 " in text
    assert "- Annotation rows over the distinct entities: 11; kept 7 " in text
    assert "Gene-entity pairs not dystonia-bearing under the primary rule: 1 of 6" in text
    single = [line for line in text.splitlines() if line.startswith("- For the neurologist")]
    assert len(single) == 1 and "GA OMIM:1 E one" in single[0] and "PANK2 OMIM:4" not in single[0]
    assert "GB OMIM:2 E two, HP:0001332 Dystonia [3/3], class combined, inheritance Autosomal recessive" in single[0]
    # contrast lines (G13)
    assert "- isolated: 5 patients, 50 units (GA 4, PANK2 1)." in text
    assert ("- Contrast (a), isolated against combined: 5 and 2 patients, k = C(5 + 2, 2) - 1 = 20 (at most "
            "1000000, enumerable), smallest p under full enumeration 0.047619 (G13;") in text
    assert "- Outside contrast (a) (G13): GB mixed (3 patients), THAP1 no_hpo_gene (3 patients)." in text
    assert ("- Contrast (b), brain iron (PANK2, PLA2G6, explicit list) against TOR1A, THAP1, GNAL: 1 and 3 "
            "patients, k = 3, smallest p 0.25 (G11, G13; exploratory).") in text
    assert "\u2014" not in text and "\u2013" not in text


def test_inheritance_reads_aspect_i_only_and_drops_not(onto):
    rows = [("HP:0000007", "", "", "I"), ("HP:0001336", "", "NOT", "I"), ("HP:0001332", "1/1", "", "P")]
    assert S.inheritance(onto, rows) == ["Autosomal recessive inheritance"]


def test_pair_counts_shared_entity():
    rows = [{"gene": g, "disease_id": d, **{c: n for c in S.ROW_COUNTS}}
            for g, d, n in (("A", "E1", 3), ("B", "E1", 3), ("B", "E2", 4))]
    by_pair, by_ent, n_distinct, shared = S.pair_counts(rows)
    assert by_pair["hpoa_rows"] == 10 and by_ent["hpoa_rows"] == 7 and n_distinct == 2
    assert shared == {"E1": ["A", "B"]}
